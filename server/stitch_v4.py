"""
Floor / slab mosaic   (v4)

Input : a Slab Capture session zip (keyframe photos + phone logs), or a video.
Output: stitched mosaic (full resolution), a preview, and result.json with
        quality metrics.

What changed from v3
  * Session input: uses the app's sharp keyframes directly (no video needed),
    and the app's live tracking as a weak link where image matching breaks
    (only where the phone reports continuous tracking between two keyframes).
  * Video input is streamed: frames are never all held in memory.
  * Loop search only considers frames that belong to the mosaic.
  * Exposure compensation works per frame footprint, not on a full canvas
    (v3 needed ~11 GB for a 20 m slab).
  * Geometry is solved on ~1024 px frames, the mosaic is rendered from the
    full-resolution keyframes (scaled homographies); output size is capped.
  * Surface-aware compositing: for rebar (things standing above the slab) each
    pixel takes its detail from the single most central view (top_k=1), so
    bars are not averaged from views that disagree; flat floors average 3.
  * Scale: mm/pixel from the app's tape footprint check, when given.

Usage
    python stitch_v4.py session_data.zip  out_dir/
    python stitch_v4.py walk.mp4          out_dir/

    from stitch_v4 import stitch
    result = stitch("session_data.zip", "out_dir")     # dict, also in out_dir/result.json

Assumes the surface is roughly flat. Tall things (walls, stairs, machines)
cannot line up and will look smeared.
"""

import csv
import io
import json
import math
import os
import time
import zipfile

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

VERSION = "4.0"

CFG = dict(
    proc_long_side=1024,     # geometry is solved on frames scaled to this long side
    step_motion=0.05,        # video input: keyframe every 5% of frame width of motion
    sharp_window=5,          # video input: sharpest frame in this many frames
    min_inliers=20,          # local pair acceptance
    loop_min_gap=8,          # keyframes apart before a pair counts as a loop
    loop_min_overlap=0.15,   # predicted ground overlap to try a loop pair
    loop_per_frame=2,        # best loop candidates tried per frame
    loop_min_hops=5,         # skip pairs already linked by < this many matched hops
    loop_gate=300,           # px: max residual drift searched for loops
    loop_min_inliers=30,
    loop_rounds=2,
    tilt_reg=5.0,            # stiffness of per-frame tilt deviation
    prior_weight=0.3,        # weight of phone-tracking links across image-matching breaks
    consensus_tau=18.0,      # grey-level gate around the per-pixel median
    top_k=None,              # detail samples per pixel; None = 1 for rebar, 3 otherwise
    render_scale=1.0,        # 1.0 = one output pixel per full-resolution keyframe pixel
    render_mode="fast",      # fast: detail from the most central view, colours blended at blend_scale
                             # blend: v3 two-band blend of top_k views at full resolution (slow)
    blend_scale=0.5,         # colour-blend pass resolution, relative to the processing canvas
    max_output_mp=80.0,      # output megapixel cap (render scale reduced to fit)
    max_full_mem_mb=1500,    # decoded full-res frames budget (else render at half size)
    preview_long_side=2048,
    jpeg_quality=92,
)

RELIEF_SURFACES = ("rebar", "wall", "other")


# =================================================================== utils
def to_h(H):
    return H / H[2, 2]


def apply(H, p):
    p = np.asarray(p, np.float64).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


def sharpness(img):
    g = cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def proc_scale(w, h, long_side):
    return min(1.0, long_side / max(w, h))


class Frame:
    """One keyframe: full-resolution JPEG bytes + processing-size image."""
    __slots__ = ("jpeg", "full_size", "img", "s", "sharp", "pose", "t_ms", "name")

    def __init__(self, jpeg, full_size, img, s, sharp, pose=None, t_ms=None, name=""):
        self.jpeg, self.full_size, self.img, self.s = jpeg, full_size, img, s
        self.sharp, self.pose, self.t_ms, self.name = sharp, pose, t_ms, name

    def full(self, reduce=1):
        flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4}[reduce]
        return cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), flag)


# ================================================================== inputs
def _csv(text):
    rows = list(csv.reader(io.StringIO(text)))
    return [dict(zip(rows[0], r)) for r in rows[1:]] if rows else []


def load_session(path, cfg, log):
    """Keyframes + phone poses from a Slab Capture export."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        meta = json.loads(z.read("session.json")) if "session.json" in names else {}
        kfs = _csv(z.read("keyframes.csv").decode()) if "keyframes.csv" in names else []
        frames_log = _csv(z.read("frames.csv").decode()) if "frames.csv" in names else []
        if not kfs:   # older export: take the images alone
            kfs = [dict(file=n) for n in sorted(names) if n.startswith("keyframes/") and n.endswith(".jpg")]
        out = []
        for r in kfs:
            if r.get("file") not in names:
                continue
            jpeg = z.read(r["file"])
            full = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if full is None:
                continue
            H, W = full.shape[:2]
            s = proc_scale(W, H, cfg["proc_long_side"])
            img = full if s == 1.0 else cv2.resize(full, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
            pose = None
            try:
                a, b, c, d, e, f = (float(r[f"pose_{k}"]) for k in "abcdef")
                pose = np.array([[a, b, c], [d, e, f], [0, 0, 1.0]])
            except (KeyError, ValueError):
                pass
            t = float(r["t_ms"]) if r.get("t_ms") not in (None, "") else None
            out.append(Frame(jpeg, (W, H), img, s, sharpness(img), pose, t, r["file"]))
    # continuous phone tracking between consecutive keyframes? (for weak links)
    valid_t = None
    if frames_log and "track_valid" in frames_log[0]:
        valid_t = np.array([[float(r["t_ms"]), float(r["track_valid"] or 0)] for r in frames_log if r.get("t_ms")])
    log(f"[1] session {meta.get('name', '?')}: {len(out)} keyframes "
        f"{out[0].full_size[0]}x{out[0].full_size[1]} -> processing {out[0].img.shape[1]}x{out[0].img.shape[0]}"
        if out else "[1] session has no keyframes")
    return out, meta, valid_t


def load_video(path, cfg, log):
    """Streamed keyframe selection: sharpest frame per motion step."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    out, prev, acc, n = [], None, 0.0, 0
    window, best = None, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        H, W = f.shape[:2]
        s = proc_scale(W, H, cfg["proc_long_side"])
        img = f if s == 1.0 else cv2.resize(f, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        sh = float(cv2.Laplacian(g, cv2.CV_64F).var())
        motion = 0.0
        if prev is not None:
            p = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8)
            if p is not None:
                q, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p, None, winSize=(21, 21), maxLevel=3)
                d = (q - p).reshape(-1, 2)[st.ravel() == 1]
                if len(d) > 10:
                    motion = float(np.linalg.norm(np.median(d, 0))) / g.shape[1]
        prev = g
        cand = (sh, f, img)
        if not out and window is None:
            window, best = 0, cand          # first frame
        else:
            acc += motion
            if window is None and acc >= cfg["step_motion"]:
                window, best = 0, None
        if window is not None:
            if best is None or cand[0] > best[0]:
                best = cand
            window += 1
            if window >= (1 if not out else cfg["sharp_window"]):
                sh_b, f_b, img_b = best
                ok_, jpg = cv2.imencode(".jpg", f_b, [cv2.IMWRITE_JPEG_QUALITY, 95])
                out.append(Frame(jpg.tobytes(), (f_b.shape[1], f_b.shape[0]), img_b,
                                 proc_scale(f_b.shape[1], f_b.shape[0], cfg["proc_long_side"]), sh_b))
                window, best, acc = None, None, 0.0
    cap.release()
    log(f"[1] video: {n} frames -> {len(out)} keyframes")
    return out, {}, None


# ================================================================ features
SIFT = cv2.SIFT_create(4000, contrastThreshold=0.01)
SIFT_DENSE = cv2.SIFT_create(4000, contrastThreshold=0.005)
CLAHE = cv2.createCLAHE(2.0, (8, 8))
BF = cv2.BFMatcher(cv2.NORM_L2)
FLANN = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))


def root_sift(des):
    if des is None:
        return None
    des = des / (des.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(des).astype(np.float32)


def features(img, mask=None, upright=False):
    g = CLAHE.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    det = SIFT_DENSE if upright else SIFT
    if upright:
        kp = det.detect(g, mask)
        for k in kp:
            k.angle = 0.0
        kp, des = det.compute(g, kp)
    else:
        kp, des = det.detectAndCompute(g, mask)
    if not kp:
        return np.zeros((0, 2), np.float32), None
    return np.float32([k.pt for k in kp]), root_sift(des)


def match_pair(fa, fb, min_inl, ratio=0.8):
    pa, da = fa
    pb, db = fb
    if da is None or db is None or len(pa) < 8 or len(pb) < 8:
        return None
    ms = [m for m, n in (x for x in FLANN.knnMatch(da, db, k=2) if len(x) == 2)
          if m.distance < ratio * n.distance]
    if len(ms) < min_inl:
        return None
    A = pa[[m.queryIdx for m in ms]]
    B = pb[[m.trainIdx for m in ms]]
    H, inl = cv2.findHomography(A, B, cv2.USAC_MAGSAC, 3.0, maxIters=5000, confidence=0.999)
    if H is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < min_inl:
        return None
    H = to_h(H)
    s = np.sqrt(abs(np.linalg.det(H[:2, :2])))
    if not (0.6 < s < 1.6):
        return None
    return dict(H=H, A=A[inl], B=B[inl], n=int(inl.sum()), kind="local")


def prior_pair(fi, fj, shape):
    """Virtual correspondences i -> j from the phone's tracking poses."""
    if fi.pose is None or fj.pose is None:
        return None
    h, w = shape[:2]
    S = min(w, h)
    rel = np.linalg.inv(fj.pose) @ fi.pose           # normalised coords i -> j
    xs, ys = np.meshgrid(np.linspace(0.12, 0.88, 3) * w, np.linspace(0.12, 0.88, 3) * h)
    A = np.c_[xs.ravel(), ys.ravel()]
    B = apply(rel, (A - [w / 2, h / 2]) / S) * S + [w / 2, h / 2]
    inside = (B[:, 0] > 0.03 * w) & (B[:, 0] < 0.97 * w) & (B[:, 1] > 0.03 * h) & (B[:, 1] < 0.97 * h)
    if inside.sum() < 4:
        return None
    return dict(A=A[inside].astype(np.float32), B=B[inside].astype(np.float32), n=int(inside.sum()), kind="prior")


# ======================================================= bundle adjustment
def rect_matrix(r, cx, cy):
    l1, l2, k, s = r
    C = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    P = np.array([[1 + k, s, 0], [0, 1, 0], [l1 * 1e-3, l2 * 1e-3, 1.0]])
    return np.linalg.inv(C) @ P @ C


def tilt_matrix(p, cx, cy):
    C = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    P = np.array([[1, 0, 0], [0, 1, 0], [p[0] * 1e-4, p[1] * 1e-4, 1.0]])
    return np.linalg.inv(C) @ P @ C


def sim_matrix(p):
    a, b, tx, ty = p
    return np.array([[1 + a, -b, tx], [b, 1 + a, ty], [0, 0, 1.0]])


KIND_WEIGHT = dict(local=1.0, loop=1.5)


def bundle_adjust(n, pairs, M_init, shape, ref, x_prev=None, tilt_reg=5.0, prior_weight=0.3,
                  max_pts=60, per_frame_tilt=True, max_nfev=100):
    """frame i -> ground = S_i @ R @ P_i ; S_ref = I."""
    h, w = shape[:2]
    cx, cy = w / 2, h / 2
    rng = np.random.default_rng(0)
    data = []
    for (i, j), r in pairs.items():
        k = min(max_pts, r['n'])
        sel = rng.choice(r['n'], k, replace=False)
        wgt = prior_weight if r['kind'] == 'prior' else KIND_WEIGHT.get(r['kind'], 1.0)
        data.append((i, j, r['A'][sel], r['B'][sel], wgt, r['kind'] != 'prior'))

    others = [i for i in range(n) if i != ref]
    col_s = {i: 4 + 4 * k for k, i in enumerate(others)}
    base_t = 4 + 4 * len(others)
    col_t = {i: base_t + 2 * i for i in range(n)}
    nparam = base_t + (2 * n if per_frame_tilt else 0)

    if x_prev is not None and len(x_prev) == nparam:
        x0 = x_prev.copy()
    else:
        x0 = np.zeros(nparam)
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        for i in others:
            M, _ = cv2.estimateAffinePartial2D(corners, apply(M_init[i], corners).astype(np.float32))
            x0[col_s[i]:col_s[i] + 4] = [M[0, 0] - 1, M[1, 0], M[0, 2], M[1, 2]]
        if x_prev is not None:
            x0[:len(x_prev)] = x_prev[:len(x0)]

    def frames_of(x):
        R = rect_matrix(x[:4], cx, cy)
        out = []
        for i in range(n):
            S = np.eye(3) if i == ref else sim_matrix(x[col_s[i]:col_s[i] + 4])
            P = tilt_matrix(x[col_t[i]:col_t[i] + 2], cx, cy) if per_frame_tilt else np.eye(3)
            out.append(S @ R @ P)
        return out

    I = np.concatenate([np.full(len(d[2]), d[0]) for d in data])
    Jf = np.concatenate([np.full(len(d[2]), d[1]) for d in data])
    PA = np.concatenate([d[2] for d in data]).astype(np.float64)
    PB = np.concatenate([d[3] for d in data]).astype(np.float64)
    WG = np.concatenate([np.full(len(d[2]), d[4]) for d in data])[:, None]
    IMG = np.concatenate([np.full(len(d[2]), d[5]) for d in data])     # image (not prior) residual
    PAh = np.c_[PA, np.ones(len(PA))]
    PBh = np.c_[PB, np.ones(len(PB))]
    Np = len(PA)

    def proj(Hs, P):
        q = np.einsum('nij,nj->ni', Hs, P)
        return q[:, :2] / q[:, 2:3]

    def resid(x):
        M = np.array(frames_of(x))
        Minv = np.linalg.inv(M)
        f = WG * (proj(Minv[Jf] @ M[I], PAh) - PB)
        b = WG * (proj(Minv[I] @ M[Jf], PBh) - PA)
        out = [f.ravel(), b.ravel()]
        if per_frame_tilt:
            out.append(tilt_reg * x[base_t:])
        return np.concatenate(out)

    def frame_cols(fr):
        c = []
        if fr != ref:
            c += list(range(col_s[fr], col_s[fr] + 4))
        if per_frame_tilt:
            c += [col_t[fr], col_t[fr] + 1]
        return c

    m = 4 * Np + (2 * n if per_frame_tilt else 0)
    J = lil_matrix((m, nparam), dtype=np.int8)
    k = 0
    for i, j, A, B, _, _ in data:
        cols = list(range(4)) + frame_cols(i) + frame_cols(j)
        L = len(A)
        J[2 * k:2 * (k + L), cols] = 1
        J[2 * Np + 2 * k:2 * Np + 2 * (k + L), cols] = 1
        k += L
    row = 4 * Np
    if per_frame_tilt:
        for q in range(2 * n):
            J[row + q, base_t + q] = 1

    res = least_squares(resid, x0, jac_sparsity=J, loss='soft_l1', f_scale=2.0,
                        x_scale='jac', max_nfev=max_nfev, ftol=1e-6, xtol=1e-6)
    r = res.fun[:row]
    err = np.linalg.norm(r.reshape(-1, 2), axis=1) / np.r_[WG.ravel(), WG.ravel()]
    img_mask = np.r_[IMG, IMG].astype(bool)
    return frames_of(res.x), res.x, err[img_mask]


def solve(n, pairs, M_init, shape, ref, cfg, x_prev=None):
    if x_prev is None:
        _, x_prev, _ = bundle_adjust(n, pairs, M_init, shape, ref, per_frame_tilt=False,
                                     prior_weight=cfg['prior_weight'], max_nfev=60)
    return bundle_adjust(n, pairs, M_init, shape, ref, x_prev=x_prev, tilt_reg=cfg['tilt_reg'],
                         prior_weight=cfg['prior_weight'])


def pair_errors(pairs, M):
    out = {}
    for (i, j), r in pairs.items():
        e = np.linalg.norm(apply(np.linalg.inv(M[j]) @ M[i], r['A']) - r['B'], axis=1)
        out[(i, j)] = float(np.median(e))
    return out


# ============================================================ loop closure
def ground_features(img):
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    m[5:-5, 5:-5] = 255
    return features(img, m, upright=True)


def guided_loop_match(img_i, feat_j, shape_j, H_ij, gate, min_inl):
    h, w = shape_j[:2]
    T = np.array([[1, 0, gate], [0, 1, gate], [0, 0, 1.0]])
    size = (w + 2 * gate, h + 2 * gate)
    full = np.full(img_i.shape[:2], 255, np.uint8)
    wi = cv2.warpPerspective(img_i, T @ H_ij, size)
    mi = cv2.erode(cv2.warpPerspective(full, T @ H_ij, size, flags=cv2.INTER_NEAREST),
                   np.ones((9, 9), np.uint8))
    mj = np.zeros(size[::-1], np.uint8)
    mj[gate:gate + h, gate:gate + w] = 255
    mi &= cv2.dilate(mj, np.ones((2 * gate // 3 * 2 + 1,) * 2, np.uint8))
    pa, da = features(wi, mi, upright=True)
    pb, db = feat_j
    if da is None or db is None or len(pa) < 20 or len(pb) < 20:
        return None
    pb = pb + gate
    good = []
    for ms in BF.knnMatch(da, db, k=8):
        c = [m for m in ms if np.linalg.norm(pa[m.queryIdx] - pb[m.trainIdx]) < gate]
        if len(c) >= 2 and c[0].distance < 0.85 * c[1].distance:
            good.append(c[0])
    if len(good) < min_inl:
        return None
    A = pa[[m.queryIdx for m in good]]
    B = pb[[m.trainIdx for m in good]]
    S, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                         maxIters=5000, confidence=0.999)
    if S is None:
        return None
    inl = inl.ravel().astype(bool)
    rot = np.degrees(np.arctan2(S[1, 0], S[0, 0]))
    sc = np.hypot(S[0, 0], S[1, 0])
    if inl.sum() < min_inl or abs(rot) > 8 or not (0.8 < sc < 1.2):
        return None
    B_in = B[inl]
    if np.sqrt(max(np.linalg.det(np.cov(B_in.T)), 0)) < 2000:
        return None
    A_orig = apply(np.linalg.inv(T @ H_ij), A[inl]).astype(np.float32)
    B_orig = (B_in - gate).astype(np.float32)
    return dict(A=A_orig, B=B_orig, n=int(inl.sum()), kind="loop", rot=rot, scale=sc)


def find_loops(imgs, M, pairs, members, cfg, log, gcache, tried_set):
    """Revisit detection among mosaic members from current ground positions."""
    h, w = imgs[0].shape[:2]
    full = np.full((h // 4, w // 4), 255, np.uint8)
    D = np.diag([4.0, 4.0, 1.0])
    Di = np.linalg.inv(D)
    ctr = {i: apply(M[i], [[w / 2, h / 2]])[0] for i in members}
    adj = {k: set() for k in members}
    for a_, b_ in pairs:
        if a_ in adj and b_ in adj:
            adj[a_].add(b_)
            adj[b_].add(a_)

    def hops_from(src, limit):
        dist = {src: 0}
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                if dist[u] >= limit:
                    continue
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        return dist

    cand = {}
    mem = sorted(members)
    for i in mem:
        near = hops_from(i, cfg['loop_min_hops'])
        for j in mem:
            if j >= i - cfg['loop_min_gap']:
                break
            if (i, j) in pairs or (i, j) in tried_set or j in near \
                    or np.linalg.norm(ctr[i] - ctr[j]) > 1.2 * max(w, h):
                continue
            Hs = Di @ np.linalg.inv(M[j]) @ M[i] @ D
            ov = (cv2.warpPerspective(full, Hs, (w // 4, h // 4), flags=cv2.INTER_NEAREST) > 0).mean()
            if ov >= cfg['loop_min_overlap']:
                cand.setdefault(i, []).append((ov, j))
    tried = added = 0
    for i, lst in cand.items():
        for ov, j in sorted(lst, reverse=True)[:cfg['loop_per_frame']]:
            tried += 1
            tried_set.add((i, j))
            if j not in gcache:
                gcache[j] = ground_features(imgs[j])
            r = guided_loop_match(imgs[i], gcache[j], imgs[j].shape,
                                  np.linalg.inv(M[j]) @ M[i], cfg['loop_gate'], cfg['loop_min_inliers'])
            if r:
                pairs[(i, j)] = r
                added += 1
    log(f"    loop search: {tried} candidates tried, {added} accepted")
    return added


# ============================================================== exposure
def gain_compensation(imgs, M, canvas_T, down=4):
    """Per-frame gains from overlap means, working inside each frame's footprint."""
    n = len(imgs)
    h, w = imgs[0].shape[:2]
    D = np.diag([1 / down, 1 / down, 1])
    corners = [[0, 0], [w, 0], [w, h], [0, h]]
    full = np.full((h, w), 255, np.uint8)
    boxes, warped, valid = [], [], []
    for i in range(n):
        T = D @ canvas_T @ M[i]
        c = apply(T, corners)
        x0, y0 = np.floor(c.min(0)).astype(int)
        x1, y1 = np.ceil(c.max(0)).astype(int)
        Tb = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]]) @ T
        size = (max(1, x1 - x0), max(1, y1 - y0))
        warped.append(cv2.warpPerspective(imgs[i], Tb, size).astype(np.float32))
        valid.append(cv2.erode(cv2.warpPerspective(full, Tb, size, flags=cv2.INTER_NEAREST),
                               np.ones((5, 5), np.uint8)) > 0)
        boxes.append((x0, y0, x1, y1))
    sn, sg = 10.0, 0.1
    A = np.zeros((3, n, n))
    b = np.zeros((3, n))
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = boxes[i], boxes[j]
            X0, Y0, X1, Y1 = max(bi[0], bj[0]), max(bi[1], bj[1]), min(bi[2], bj[2]), min(bi[3], bj[3])
            if X1 - X0 < 4 or Y1 - Y0 < 4:
                continue
            si = (slice(Y0 - bi[1], Y1 - bi[1]), slice(X0 - bi[0], X1 - bi[0]))
            sj = (slice(Y0 - bj[1], Y1 - bj[1]), slice(X0 - bj[0], X1 - bj[0]))
            ov = valid[i][si] & valid[j][sj]
            N = int(ov.sum())
            if N < 200:
                continue
            Ii = warped[i][si][ov].mean(0)
            Ij = warped[j][sj][ov].mean(0)
            for c in range(3):
                A[c, i, i] += N * Ii[c] ** 2 / sn ** 2 + N / sg ** 2
                A[c, j, j] += N * Ij[c] ** 2 / sn ** 2 + N / sg ** 2
                A[c, i, j] -= N * Ii[c] * Ij[c] / sn ** 2
                A[c, j, i] -= N * Ii[c] * Ij[c] / sn ** 2
                b[c, i] += N / sg ** 2
                b[c, j] += N / sg ** 2
    gains = np.ones((n, 3))
    for c in range(3):
        for i in range(n):
            if A[c, i, i] == 0:
                A[c, i, i], b[c, i] = 1, 1
        gains[:, c] = np.linalg.solve(A[c], b[c])
    return gains


# ============================================================ compositing
def composite(imgs, M, sharp, gains, canvas_T, size, tau, top_k, tile=384, pad=48,
              low_sigma=10.0, progress=None, parts=False):
    """Two-band robust compositing (see v3); M maps each image to the canvas.
    parts=True also returns the blended low band (float32) and, per pixel, the
    index of the most central consistent view (-1 = empty)."""
    Wc, Hc = size
    out = np.zeros((Hc, Wc, 3), np.uint8)
    cover = np.zeros((Hc, Wc), bool)
    if parts:
        low_out = np.zeros((Hc, Wc, 3), np.float32)
        label = np.full((Hc, Wc), -1, np.int32)
    feathers, boxes = [], []
    for img, m in zip(imgs, M):
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        f = np.minimum(np.minimum(xx + 1, w - xx), np.minimum(yy + 1, h - yy))
        feathers.append((f / f.max()).astype(np.float32))
        c = apply(canvas_T @ m, [[0, 0], [w, 0], [w, h], [0, h]])
        boxes.append((*c.min(0), *c.max(0)))
    sw = np.asarray(sharp, np.float32)
    sw = sw / max(sw.max(), 1e-6)

    def blur(a):
        return cv2.GaussianBlur(a, (0, 0), low_sigma)

    tiles = [(x0, y0) for y0 in range(0, Hc, tile) for x0 in range(0, Wc, tile)]
    for ti, (x0, y0) in enumerate(tiles):
        if progress and ti % 4 == 0:
            progress(ti / len(tiles))
        x1, y1 = min(Wc, x0 + tile), min(Hc, y0 + tile)
        X0, Y0, X1, Y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        ids = [i for i, b in enumerate(boxes) if b[0] < X1 and b[2] > X0 and b[1] < Y1 and b[3] > Y0]
        if not ids:
            continue
        tw, th = X1 - X0, Y1 - Y0
        Tt = np.array([[1, 0, -X0], [0, 1, -Y0], [0, 0, 1.0]])
        S, Wt = [], []
        for i in ids:
            T = Tt @ canvas_T @ M[i]
            S.append(cv2.warpPerspective(imgs[i], T, (tw, th), flags=cv2.INTER_LINEAR).astype(np.float32)
                     * gains[i][None, None, :])
            Wt.append(cv2.warpPerspective(feathers[i], T, (tw, th)) * (0.5 + 0.5 * sw[i]))
        S = np.stack(S)
        Wt = np.stack(Wt)
        valid = Wt > 1e-3
        if not valid[:, pad:pad + (y1 - y0), pad:pad + (x1 - x0)].any():
            continue
        lum = S.mean(axis=3)
        with np.errstate(all='ignore'):
            med = np.nanmedian(np.where(valid, lum, np.nan), axis=0)
        inl = valid & (np.abs(lum - med[None]) < tau)
        inl = np.where((valid.sum(0) < 3)[None], valid, inl)
        none = valid.any(0) & ~inl.any(0)
        if none.any():
            dist = np.where(valid, np.abs(lum - np.nan_to_num(med)[None]), np.inf)
            pick = np.zeros_like(inl)
            np.put_along_axis(pick, np.argmin(dist, 0)[None], True, axis=0)
            inl = np.where(none[None], pick & valid, inl)
        low = np.empty_like(S)
        for k in range(len(ids)):
            v = valid[k].astype(np.float32)
            nv = blur(v)[..., None]
            low[k] = blur(S[k] * v[..., None]) / np.maximum(nv, 1e-4)
        high = S - low
        Wl = np.where(inl, Wt, 0)
        wls = Wl.sum(0)
        L = (low * Wl[..., None]).sum(0) / np.maximum(wls, 1e-6)[..., None]
        Wh = np.where(inl, Wt ** 3, 0)
        if len(ids) > top_k:
            thr = -np.sort(-Wh, axis=0)[top_k - 1]
            Wh = np.where(Wh >= thr[None], Wh, 0)
        whs = Wh.sum(0)
        Hb = (high * Wh[..., None]).sum(0) / np.maximum(whs, 1e-9)[..., None]
        res = L + Hb
        ok = wls > 1e-6
        sl = (slice(pad, pad + (y1 - y0)), slice(pad, pad + (x1 - x0)))
        okc = ok[sl]
        out[y0:y1, x0:x1][okc] = np.clip(res[sl][okc], 0, 255).astype(np.uint8)
        cover[y0:y1, x0:x1] |= okc
        if parts:
            low_out[y0:y1, x0:x1] = L[sl]
            best = np.asarray(ids)[np.argmax(np.where(inl, Wt ** 3, -1), axis=0)]
            label[y0:y1, x0:x1] = np.where(ok, best, -1)[sl]
    if parts:
        return out, cover, low_out, label
    return out, cover


def render_fast(frames_full, M_out, gains, canvas_T, size, low, label, blend_scale, low_sigma, progress=None):
    """Full-resolution render: colours / exposure = low band blended at
    `blend_scale` of the output, detail = high band of the single most central
    consistent view per pixel (label map from the blend pass)."""
    Wc, Hc = size
    # remove single-pixel islands in the view choice (they speckle on non-flat things)
    label = cv2.medianBlur(label.astype(np.float32), 5).astype(np.int32)
    low_up = cv2.resize(low, (Wc, Hc), interpolation=cv2.INTER_LINEAR)
    lab = cv2.resize(label.astype(np.float32), (Wc, Hc), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    # start from the blended colours, so any pixel the chosen view cannot fill is not black
    out = np.clip(low_up, 0, 255).astype(np.uint8)
    out[lab < 0] = 0
    marg = int(3 * low_sigma) + 2
    for k, (img, m) in enumerate(zip(frames_full, M_out)):
        if progress and k % 3 == 0:
            progress(k / len(frames_full))
        ys, xs = np.nonzero(label == k)
        if not len(ys):
            continue
        x0 = max(0, int(xs.min() / blend_scale) - marg)
        x1 = min(Wc, int((xs.max() + 1) / blend_scale) + marg)
        y0 = max(0, int(ys.min() / blend_scale) - marg)
        y1 = min(Hc, int((ys.max() + 1) / blend_scale) + marg)
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]]) @ canvas_T @ m
        w_, h_ = x1 - x0, y1 - y0
        warped = cv2.warpPerspective(img, T, (w_, h_), flags=cv2.INTER_LINEAR).astype(np.float32) \
            * gains[k][None, None, :].astype(np.float32)
        valid = cv2.warpPerspective(np.ones(img.shape[:2], np.uint8), T, (w_, h_),
                                    flags=cv2.INTER_NEAREST).astype(np.float32)
        nv = cv2.GaussianBlur(valid, (0, 0), low_sigma)[..., None]
        lowk = cv2.GaussianBlur(warped * valid[..., None], (0, 0), low_sigma) / np.maximum(nv, 1e-4)
        sel = (lab[y0:y1, x0:x1] == k) & (valid > 0)
        res = low_up[y0:y1, x0:x1] + (warped - lowk)
        out[y0:y1, x0:x1][sel] = np.clip(res[sel], 0, 255).astype(np.uint8)
    return out, lab >= 0


# =================================================================== main
def stitch(src, out_dir, cfg=None, log=print, progress=None):
    """src: session .zip or video. Writes mosaic.jpg, preview.jpg, result.json into out_dir."""
    cfg = {**CFG, **(cfg or {})}
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    _log = log
    log = lambda m: _log(f"{time.time() - t0:6.0f}s  {m}")
    prog = progress or (lambda stage, frac: None)
    warnings = []

    prog("reading", 0.0)
    if str(src).lower().endswith(".zip"):
        frames, meta, valid_t = load_session(src, cfg, log)
    else:
        frames, meta, valid_t = load_video(src, cfg, log)
    n = len(frames)
    if n < 2:
        raise RuntimeError("need at least 2 keyframes")
    imgs = [f.img for f in frames]
    shapes = {im.shape[:2] for im in imgs}
    if len(shapes) > 1:           # phone rotated mid-capture: keep the dominant orientation
        dom = max(shapes, key=lambda s: sum(im.shape[:2] == s for im in imgs))
        keep_idx = [i for i, im in enumerate(imgs) if im.shape[:2] == dom]
        warnings.append(f"{n - len(keep_idx)} keyframes had a different orientation (phone rotated) and were skipped")
        frames = [frames[i] for i in keep_idx]
        imgs = [f.img for f in frames]
        n = len(frames)
    h, w = imgs[0].shape[:2]
    surface = str(meta.get("surface", "")).lower()
    top_k = cfg['top_k'] or (1 if any(s in surface for s in RELIEF_SURFACES) else 3)

    prog("features", 0.05)
    F = []
    for i, im in enumerate(imgs):
        F.append(features(im))
        if i % 5 == 0:
            prog("features", 0.05 + 0.2 * i / n)

    # ---- local pairs; initial poses from chained similarities
    prog("matching", 0.25)
    pairs, S, breaks, weak = {}, [np.eye(3)], [], []
    for i in range(1, n):
        got = False
        for back in range(1, 9):
            if i - back < 0 or (back > 3 and got):
                break
            r = match_pair(F[i], F[i - back], cfg['min_inliers'])
            if r:
                pairs[(i, i - back)] = r
                if not got:
                    A, _ = cv2.estimateAffinePartial2D(r['A'], r['B'], method=cv2.RANSAC,
                                                       ransacReprojThreshold=3.0)
                    S.append(S[i - back] @ np.vstack([A, [0, 0, 1]]))
                    got = True
        if not got:
            # weak link from the phone's tracking, only if it tracked continuously in between
            r = None
            if valid_t is not None and frames[i].t_ms is not None and frames[i - 1].t_ms is not None:
                seg = valid_t[(valid_t[:, 0] >= frames[i - 1].t_ms) & (valid_t[:, 0] <= frames[i].t_ms)]
                if len(seg) and seg[:, 1].min() > 0:
                    r = prior_pair(frames[i], frames[i - 1], imgs[i].shape)
            if r:
                pairs[(i, i - 1)] = r
                weak.append(i)
                A, _ = cv2.estimateAffinePartial2D(r['A'], r['B'])
                S.append(S[i - 1] @ np.vstack([A, [0, 0, 1]]))
            else:
                breaks.append(i)
                S.append(S[-1].copy())
        if i % 5 == 0:
            prog("matching", 0.25 + 0.2 * i / n)
    log(f"[2] local pairs: {sum(1 for p in pairs.values() if p['kind'] == 'local')}"
        + (f", phone-tracking links at {weak}" if weak else "")
        + (f", breaks at {breaks}" if breaks else ""))

    # ---- largest connected group
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in pairs:
        parent[root(i)] = root(j)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    main = max(groups.values(), key=len)
    if len(main) < n:
        warnings.append(f"{n - len(main)} of {n} keyframes could not be connected to the mosaic "
                        f"(tracking lost) and were skipped")
        log("    WARNING: " + warnings[-1])
    if len(main) < 2:
        raise RuntimeError("no two keyframes could be matched – surface too plain, too fast, or too blurry")
    members = set(main)
    pairs = {k: v for k, v in pairs.items() if k[0] in members and k[1] in members}
    ref = main[len(main) // 2]
    M0 = [np.linalg.inv(S[ref]) @ Si for Si in S]

    # ---- BA + loop closure
    prog("aligning", 0.45)
    M, x, err = solve(n, pairs, M0, imgs[0].shape, ref, cfg)
    log(f"[3] BA: median {np.median(err):.2f}px  p90 {np.percentile(err, 90):.2f}px")
    gcache, tried_set, n_loops = {}, set(), 0
    for rnd in range(cfg['loop_rounds']):
        prog("loop closure", 0.5 + 0.1 * rnd)
        added = find_loops(imgs, M, pairs, members, cfg, log, gcache, tried_set)
        if not added:
            break
        n_loops += added
        M, x, err = solve(n, pairs, M, imgs[0].shape, ref, cfg, x_prev=x)
        log(f"[4.{rnd + 1}] BA with {sum(1 for p in pairs.values() if p['kind'] == 'loop')} loops: "
            f"median {np.median(err):.2f}px  p90 {np.percentile(err, 90):.2f}px")
    pe = pair_errors(pairs, M)
    bad = [k for k, v in pe.items() if v > 6.0 and pairs[k]['kind'] != 'prior']
    if bad:
        for k in bad:
            pairs.pop(k)
        M, x, err = solve(n, pairs, M, imgs[0].shape, ref, cfg, x_prev=x)
        log(f"[5] removed {len(bad)} inconsistent pairs -> median {np.median(err):.2f}px "
            f"p90 {np.percentile(err, 90):.2f}px")
    used = sorted({i for k in pairs for i in k} & members)
    q_med, q_p90 = float(np.median(err)), float(np.percentile(err, 90))
    quality = "good" if q_med <= 1.5 and q_p90 <= 4 else "ok" if q_med <= 2.5 and q_p90 <= 7 else "poor"
    if quality == "poor":
        warnings.append(f"alignment is poor (median {q_med:.1f}px, p90 {q_p90:.1f}px): expect visible misalignment")

    # ---- orientation: main walking direction horizontal
    c = np.array([apply(M[i], [[w / 2, h / 2]])[0] for i in used])
    if len(c) >= 3:
        _, _, Vt = np.linalg.svd(c - c.mean(0), full_matrices=False)
        a = -np.arctan2(Vt[0, 1], Vt[0, 0])
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
        M = [Rz @ m for m in M]

    # ---- canvas at processing scale, then output scale
    corners = [[0, 0], [w, 0], [w, h], [0, h]]
    allc = np.vstack([apply(M[i], corners) for i in used])
    mn, mx = allc.min(0), allc.max(0)
    T = np.array([[1, 0, -mn[0] + 2], [0, 1, -mn[1] + 2], [0, 0, 1.0]])
    size_p = (int(np.ceil(mx[0] - mn[0])) + 4, int(np.ceil(mx[1] - mn[1])) + 4)

    s_proc = frames[used[0]].s                      # full -> processing scale
    g = cfg['render_scale'] / s_proc                # processing canvas -> output canvas
    if size_p[0] * size_p[1] * g * g > cfg['max_output_mp'] * 1e6:
        g = math.sqrt(cfg['max_output_mp'] * 1e6 / (size_p[0] * size_p[1]))
        warnings.append(f"output limited to {cfg['max_output_mp']:.0f} MP")
    reduce = 1
    full_mb = sum(f.full_size[0] * f.full_size[1] * 3 for f in (frames[i] for i in used)) / 1e6
    if g * s_proc <= 0.55 or full_mb > cfg['max_full_mem_mb']:
        reduce = 2 if (g * s_proc <= 0.55 or full_mb / 4 <= cfg['max_full_mem_mb']) else 4
    G = np.diag([g, g, 1.0])
    size = (int(math.ceil(size_p[0] * g)), int(math.ceil(size_p[1] * g)))
    log(f"[6] canvas {size[0]}x{size[1]} ({size[0] * size[1] / 1e6:.1f} MP), render from "
        f"{'full' if reduce == 1 else f'1/{reduce}'}-resolution keyframes")

    # exposure (processing scale)
    prog("exposure", 0.62)
    imgs_u = [imgs[i] for i in used]
    M_u = [M[i] for i in used]
    gains = gain_compensation(imgs_u, M_u, T)

    # render images: full-res pixel -> processing pixel -> ground -> output
    prog("rendering", 0.65)
    rimgs, rM = [], []
    for i in used:
        im = frames[i].full(reduce)
        rimgs.append(im)
        rM.append(G @ M[i] @ np.diag([imgs[i].shape[1] / im.shape[1], imgs[i].shape[0] / im.shape[0], 1.0]))
    rT = G @ T @ np.linalg.inv(G)
    if cfg['render_mode'] == 'fast':
        # 1) colour blend + best-view labels at low resolution
        q = cfg['blend_scale']
        Q = np.diag([q, q, 1.0])
        bimgs, bM = [], []
        for i in used:
            b = cv2.resize(imgs[i], None, fx=q, fy=q, interpolation=cv2.INTER_AREA) if q < 1 else imgs[i]
            bimgs.append(b)
            bM.append(Q @ M[i] @ np.diag([imgs[i].shape[1] / b.shape[1], imgs[i].shape[0] / b.shape[0], 1.0]))
        bT = Q @ T @ np.linalg.inv(Q)
        bsize = (int(math.ceil(size_p[0] * q)), int(math.ceil(size_p[1] * q)))
        _, _, low, label = composite(bimgs, bM, [frames[i].sharp for i in used], gains, bT, bsize,
                                     cfg['consensus_tau'], 1, tile=256, pad=24, low_sigma=10.0 * q,
                                     progress=lambda f: prog("rendering", 0.65 + 0.12 * f), parts=True)
        del bimgs
        # 2) full-resolution detail from the chosen view
        pano, cover = render_fast(rimgs, rM, gains, rT, size, low, label, q / g, 10.0 * g,
                                  progress=lambda f: prog("rendering", 0.77 + 0.19 * f))
        del low, label
    else:
        pano, cover = composite(rimgs, rM, [frames[i].sharp for i in used], gains, rT, size,
                                cfg['consensus_tau'], top_k, pad=int(48 * max(1.0, g)),
                                low_sigma=10.0 * max(1.0, g),
                                progress=lambda f: prog("rendering", 0.65 + 0.3 * f))
    del rimgs
    ys, xs = np.nonzero(cover)
    pano = pano[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    coverage = float(cover[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean())

    # ---- scale from the tape footprint check (approximate)
    mm_per_px = None
    tape = meta.get("tape_short_side_m")
    if tape:
        fw, fh = frames[used[0]].full_size
        frame_mm_px = float(tape) * 1000 / min(fw, fh)
        sc = []
        for i in used:
            Jm = (G @ M[i])
            p0 = apply(Jm, [[w / 2, h / 2]])[0]
            px = apply(Jm, [[w / 2 + 1, h / 2]])[0] - p0
            py = apply(Jm, [[w / 2, h / 2 + 1]])[0] - p0
            sc.append(math.sqrt(abs(px[0] * py[1] - px[1] * py[0])) * s_proc)   # output px per full-res px
        mm_per_px = frame_mm_px / float(np.median(sc))

    # ---- write
    prog("saving", 0.96)
    mosaic_path = os.path.join(out_dir, "mosaic.jpg")
    cv2.imwrite(mosaic_path, pano, [cv2.IMWRITE_JPEG_QUALITY, cfg['jpeg_quality']])
    ph, pw = pano.shape[:2]
    ps = min(1.0, cfg['preview_long_side'] / max(ph, pw))
    prev = cv2.resize(pano, (max(1, round(pw * ps)), max(1, round(ph * ps))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, "preview.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 85])

    result = dict(
        version=VERSION, source=os.path.basename(str(src)), session=meta.get("name"), surface=meta.get("surface"),
        keyframes_total=len(frames), keyframes_used=len(used), chain_breaks=len(breaks),
        phone_tracking_links=len(weak), loop_closures=sum(1 for p in pairs.values() if p['kind'] == 'loop'),
        ba_median_px=round(q_med, 3), ba_p90_px=round(q_p90, 3), quality=quality,
        mosaic_size=[pw, ph], mosaic_mp=round(pw * ph / 1e6, 1), coverage_of_bbox=round(coverage, 3),
        mm_per_px=round(mm_per_px, 3) if mm_per_px else None,
        mosaic_extent_m=[round(pw * mm_per_px / 1000, 2), round(ph * mm_per_px / 1000, 2)] if mm_per_px else None,
        top_k=top_k if cfg['render_mode'] == 'blend' else 1, render_mode=cfg['render_mode'], render_reduce=reduce, warnings=warnings, seconds=round(time.time() - t0, 1),
    )
    with open(os.path.join(out_dir, "result.json"), "w") as fp:
        json.dump(result, fp, indent=2)
    log(f"[7] saved {mosaic_path}  {pw}x{ph}  quality={quality}  total {time.time() - t0:.0f}s")
    prog("done", 1.0)
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python stitch_v4.py <session_data.zip | video.mp4> [out_dir]")
    r = stitch(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "mosaic_out")
    print(json.dumps(r, indent=2))
