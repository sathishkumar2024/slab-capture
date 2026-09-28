"""
Load a Slab Capture session export (the *_data.zip from the phone app).

    python load_session.py capture_data.zip                  # quality report
    python load_session.py capture_data.zip --extract out/   # unzip keyframes + logs
    python load_session.py capture_data.zip --video capture.mp4 --stitch mosaic.jpg
                                                              # run stitch_v3 on the video

    from load_session import load
    s = load("capture_data.zip")
    s.meta                  # session.json (device, camera settings, notes, summary)
    s.frames                # dict of numpy arrays, one per column of frames.csv
    s.keyframes             # list of dicts: index, file, t_ms, pose (3x3), tilt_deg, ...
    s.keyframe_image(0)     # BGR image of keyframe 0
    s.imu, s.orientation    # dicts of numpy arrays (t_ms relative to record start)

Coordinates: poses map centred frame coordinates, in units of the frame's
short side, to the first recorded frame (the app's live tracker; approximate).
Timestamps: t_ms is milliseconds since REC was pressed, shared by all logs;
media_time_s is the camera frame's own timestamp.
"""
import csv
import io
import json
import sys
import zipfile
from dataclasses import dataclass, field

import numpy as np


def _table(text):
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return {}
    head, body = rows[0], rows[1:]
    out = {}
    for j, name in enumerate(head):
        col = [r[j] if j < len(r) else "" for r in body]
        try:
            out[name] = np.array([float(v) if v != "" else np.nan for v in col])
        except ValueError:
            out[name] = np.array(col, dtype=object)
    return out


@dataclass
class Session:
    path: str
    meta: dict
    frames: dict
    imu: dict
    orientation: dict
    events: list
    keyframes: list = field(default_factory=list)

    def keyframe_image(self, i):
        import cv2
        with zipfile.ZipFile(self.path) as z:
            buf = np.frombuffer(z.read(self.keyframes[i]["file"]), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def report(self):
        m, s = self.meta, self.meta.get("summary", {})
        cam = m.get("camera", {}).get("settings", {})
        fr = self.frames
        lines = [
            f"session      {m.get('name')}  ({m.get('id')})  status={m.get('status')}",
            f"site/surface {m.get('site') or '-'} / {m.get('surface')}  lighting={m.get('lighting')}  "
            f"height={m.get('phone_height_m')} m  tape_short_side={m.get('tape_short_side_m')} m",
            f"camera       {cam.get('width')}x{cam.get('height')} @ {cam.get('frameRate')} fps  "
            f"label='{m.get('camera', {}).get('label', '')}'",
            f"device       {m.get('device', {}).get('user_agent', '')[:110]}",
            f"duration     {m.get('duration_s')} s   processed {s.get('frames_processed')} frames "
            f"@ {s.get('processing_fps')} fps",
            f"keyframes    {len(self.keyframes)}   video: {m.get('video', {}).get('mime')} "
            f"{(m.get('video', {}).get('bytes') or 0) / 1e6:.0f} MB",
            f"quality      tracking {s.get('tracking_ok_pct')}%  level {s.get('tilt_ok_pct')}%  "
            f"speed {s.get('speed_ok_pct')}%  blurred {s.get('blurred_frames_pct')}%  "
            f"red warnings {s.get('red_warnings')}",
            f"gyro         available={s.get('gyro_available')}  factor={s.get('gyro_factor')}",
            f"imu          {len(self.imu.get('t_ms', []))} samples   orientation "
            f"{len(self.orientation.get('t_ms', []))} samples",
        ]
        if len(fr.get("speed", [])):
            sp = fr["speed"]
            lines.append("speed        p50 %.2f  p90 %.2f  (frame short-sides / s)" % tuple(np.nanpercentile(sp, [50, 90])))
        if len(fr.get("tilt_deg", [])) and np.isfinite(fr["tilt_deg"]).any():
            lines.append("tilt         p50 %.1f°  p90 %.1f°" % tuple(np.nanpercentile(fr["tilt_deg"], [50, 90])))
        bad = [e for e in self.events if e[1] not in ("status",)]
        if bad:
            lines.append("events       " + "; ".join(f"{float(e[0]) / 1000:.1f}s {e[1]} {e[2]}".strip() for e in bad[:12]))
        return "\n".join(lines)


def load(path):
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            raise IOError(f"corrupt entry in zip: {bad}")
        meta = json.loads(z.read("session.json"))
        frames = _table(z.read("frames.csv").decode())
        imu = _table(z.read("imu.csv").decode())
        ori = _table(z.read("orientation.csv").decode())
        events = list(csv.reader(io.StringIO(z.read("events.csv").decode())))[1:]
        kt = _table(z.read("keyframes.csv").decode())
        names = set(z.namelist())
    kfs = []
    for i in range(len(kt.get("index", []))):
        P = [kt[f"pose_{c}"][i] for c in "abcdef"]
        kf = dict(index=int(kt["index"][i]), file=kt["file"][i], t_ms=kt["t_ms"][i],
                  media_time_s=kt["media_time_s"][i], size=(int(kt["frame_w"][i]), int(kt["frame_h"][i])),
                  pose=np.array([[P[0], P[1], P[2]], [P[3], P[4], P[5]], [0, 0, 1.0]]),
                  tilt_deg=kt["tilt_deg"][i], beta=kt["beta"][i], gamma=kt["gamma"][i],
                  sharpness=kt["sharpness"][i], speed=kt["speed"][i], track_valid=bool(kt["track_valid"][i]))
        if kf["file"] in names:
            kfs.append(kf)
    return Session(path, meta, frames, imu, ori, events, kfs)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip")
    ap.add_argument("--extract", metavar="DIR")
    ap.add_argument("--video", help="the session's video file (for --stitch)")
    ap.add_argument("--stitch", metavar="OUT_JPG", help="run stitch_v3 on --video")
    a = ap.parse_args(argv)
    s = load(a.zip)
    print(s.report())
    if a.extract:
        with zipfile.ZipFile(a.zip) as z:
            z.extractall(a.extract)
        print(f"extracted to {a.extract}")
    if a.stitch:
        if not a.video:
            sys.exit("--stitch needs --video")
        from stitch_v3 import stitch
        _, info = stitch(a.video, a.stitch)
        print(f"stitched: median {info['ba_median_px']:.2f}px  p90 {info['ba_p90_px']:.2f}px -> {a.stitch}")


if __name__ == "__main__":
    main(sys.argv[1:])
