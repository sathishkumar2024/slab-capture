"""
Slab Capture stitching server.

    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints (all under /api):
    GET  /api/health                      -> {"ok": true, "version": ...}
    POST /api/jobs        (multipart file=<session_data.zip | video>) -> {"id": ..., "status": "queued"}
    GET  /api/jobs/{id}                   -> status, stage, progress 0..1, log tail, result metrics
    GET  /api/jobs/{id}/preview.jpg       -> preview (long side <= 2048 px)
    GET  /api/jobs/{id}/mosaic.jpg        -> full-resolution mosaic
    GET  /api/jobs/{id}/result.json

Jobs run one at a time in a background thread (stitching uses all CPU cores).
Files live in $JOBS_DIR (default ./jobs); jobs older than $JOB_TTL_HOURS (default 72) are deleted.
Set $API_KEY to require the header  X-Api-Key: <key>  on job endpoints.
"""
import json
import os
import queue
import shutil
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

import stitch_v4

JOBS_DIR = os.path.abspath(os.environ.get("JOBS_DIR", "jobs"))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "1500"))
JOB_TTL_H = float(os.environ.get("JOB_TTL_HOURS", "72"))
API_KEY = os.environ.get("API_KEY", "")
os.makedirs(JOBS_DIR, exist_ok=True)

app = FastAPI(title="Slab Capture stitcher", version=stitch_v4.VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["*"])

_jobs = {}                 # id -> dict (in memory; mirrored to jobs/<id>/job.json)
_lock = threading.Lock()
_q = queue.Queue()


def _job_path(jid, *p):
    return os.path.join(JOBS_DIR, jid, *p)


def _save(job):
    with open(_job_path(job["id"], "job.json"), "w") as f:
        json.dump(job, f)


def _update(jid, **kw):
    with _lock:
        job = _jobs[jid]
        job.update(kw)
        _save(job)


def _load_existing():
    for jid in os.listdir(JOBS_DIR):
        p = _job_path(jid, "job.json")
        if os.path.isfile(p):
            try:
                job = json.load(open(p))
            except Exception:
                continue
            if job.get("status") in ("queued", "running"):
                job.update(status="error", error="server restarted while this job was running – submit again")
            _jobs[jid] = job


def _cleanup():
    cutoff = time.time() - JOB_TTL_H * 3600
    for jid, job in list(_jobs.items()):
        if job.get("created", 0) < cutoff and job.get("status") not in ("queued", "running"):
            shutil.rmtree(_job_path(jid), ignore_errors=True)
            _jobs.pop(jid, None)


def _worker():
    while True:
        jid = _q.get()
        job = _jobs.get(jid)
        if not job:
            continue
        _update(jid, status="running", started=time.time(), stage="starting", progress=0.0)
        lines = []

        def log(m):
            lines.append(str(m))
            _update(jid, log=lines[-12:])

        def prog(stage, frac):
            _update(jid, stage=stage, progress=round(float(frac), 3))

        try:
            res = stitch_v4.stitch(_job_path(jid, job["input"]), _job_path(jid, "out"), log=log, progress=prog)
            res["source"] = job.get("filename")
            _update(jid, status="done", stage="done", progress=1.0, result=res, finished=time.time())
        except Exception as e:
            traceback.print_exc()
            _update(jid, status="error", error=f"{type(e).__name__}: {e}", finished=time.time())
        finally:
            try:
                os.remove(_job_path(jid, job["input"]))      # keep outputs, drop the upload
            except OSError:
                pass


_load_existing()
threading.Thread(target=_worker, daemon=True).start()


def _auth(key):
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "missing or wrong X-Api-Key")


def _public(job):
    out = {k: job.get(k) for k in ("id", "status", "stage", "progress", "error", "result", "created",
                                   "started", "finished", "log", "filename", "size_mb")}
    if job.get("status") == "queued":
        ahead = [j for j in _jobs.values() if j.get("status") in ("queued", "running")
                 and j.get("created", 0) < job.get("created", 0)]
        out["queue_position"] = len(ahead)
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    return ("<h3>Slab Capture stitcher</h3><p>Running. Use this address as the <b>Stitch server</b> "
            "in the Slab Capture app. API: <code>/api/health</code>, <code>/api/jobs</code>.</p>")


@app.get("/api/health")
def health():
    return {"ok": True, "version": stitch_v4.VERSION, "queued": _q.qsize(),
            "auth": bool(API_KEY)}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    _auth(x_api_key)
    _cleanup()
    name = os.path.basename(file.filename or "upload.zip")
    ext = os.path.splitext(name)[1].lower()
    if ext not in (".zip", ".mp4", ".mov", ".webm", ".m4v"):
        raise HTTPException(400, "upload a session .zip (Save data) or a video file")
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    os.makedirs(_job_path(jid), exist_ok=True)
    dst = _job_path(jid, "input" + ext)
    size = 0
    with open(dst, "wb") as f:
        while True:
            chunk = await file.read(4 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1e6:
                f.close()
                shutil.rmtree(_job_path(jid), ignore_errors=True)
                raise HTTPException(413, f"upload larger than {MAX_UPLOAD_MB:.0f} MB")
            f.write(chunk)
    job = dict(id=jid, status="queued", stage="queued", progress=0.0, created=time.time(),
               input="input" + ext, filename=name, size_mb=round(size / 1e6, 1), log=[])
    with _lock:
        _jobs[jid] = job
        _save(job)
    _q.put(jid)
    return _public(job)


@app.get("/api/jobs/{jid}")
def get_job(jid: str, x_api_key: str = Header(default="")):
    _auth(x_api_key)
    job = _jobs.get(jid)
    if not job:
        raise HTTPException(404, "job not found (it may have expired)")
    return _public(job)


def _file(jid, name, media):
    p = _job_path(jid, "out", name)
    if jid not in _jobs or not os.path.isfile(p):
        raise HTTPException(404, "not ready")
    return FileResponse(p, media_type=media, filename=f"{jid}_{name}")


@app.get("/api/jobs/{jid}/preview.jpg")
def preview(jid: str):
    return _file(jid, "preview.jpg", "image/jpeg")


@app.get("/api/jobs/{jid}/mosaic.jpg")
def mosaic(jid: str):
    return _file(jid, "mosaic.jpg", "image/jpeg")


@app.get("/api/jobs/{jid}/result.json")
def result(jid: str):
    return _file(jid, "result.json", "application/json")
