---
title: Slab Capture Stitcher
emoji: 🧱
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Slab Capture stitching server (v4)

The phone app uploads a capture (keyframe photos and logs, not the video). This server stitches it and the app shows the result.
It is `stitch_v4.py` wrapped in a small FastAPI service.

The app runs on an `https://` page, so **the server must also be reachable over `https://`**. Choose A or B.

## A. Your laptop + a free tunnel (fastest to start)

```bash
cd ~/Documents/slab-capture && git pull      # get the latest code
cd server
./run_local.sh                               # first run installs packages (~2 min)
```
If it says `ensurepip is not available`, run `sudo apt install python3-venv` first.

In a **second terminal**, install cloudflared once:
```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```
Then open the tunnel:
```bash
cloudflared tunnel --url http://localhost:8000
```
It prints a line like `https://random-words.trycloudflare.com`. In the app, go to **start page → Stitching server**, paste that address and tap **Test connection**. It should say "Connected ✓".

Keep in mind:
- Keep the laptop awake with both terminals open while you stitch. The phone can be anywhere with mobile data.
- The tunnel address changes every time you restart `cloudflared`, so paste the new one into the app.
- A free quick tunnel is for testing, not production. Keep captures to 1–3 minutes so uploads stay small (a typical upload is 5–40 MB).

## B. Hugging Face Space (always on, free CPU)

1. On huggingface.co: **New Space** → name `slab-stitcher` → SDK **Docker** (Blank) → hardware **CPU basic (free)** → **Public** → Create.
2. **Files → Add file → Upload files**. Upload this folder's `Dockerfile`, `app.py`, `stitch_v4.py`, `requirements.txt` and `README.md` (this file; its header tells the Space to use Docker on port 7860). Commit.
3. Wait for the build (3–5 min, status "Running").
4. Add a secret so strangers can't use your server: **Settings → Variables and secrets → New secret** `API_KEY` = any password. Enter the same value in the app as **Access key**.
5. The server address is `https://<your-hf-username>-slab-stitcher.hf.space`.

Free Spaces sleep after about two days without use (the first request wakes them up in about a minute), and their files are wiped on restart. Save stitched images you want to keep.

## C. Docker anywhere

```bash
docker build -t slab-stitcher server
docker run -p 8000:7860 -e API_KEY=your-secret slab-stitcher
```

## API

| Method | Path | |
|---|---|---|
| GET | `/api/health` | `{"ok": true, "version": "4.0"}` |
| POST | `/api/jobs` | multipart `file` = session `.zip` (Save data) or a video. Returns `{"id", "status"}` |
| GET | `/api/jobs/{id}` | status (`queued` / `running` / `done` / `error`), stage, progress 0–1, log, result metrics |
| GET | `/api/jobs/{id}/preview.jpg` | preview, long side ≤ 2048 px |
| GET | `/api/jobs/{id}/mosaic.jpg` | full-resolution mosaic |
| GET | `/api/jobs/{id}/result.json` | metrics (alignment error, photos used, loops, mm/px, warnings) |

- Jobs run one at a time and are deleted after 72 h (`JOB_TTL_HOURS`).
- With `API_KEY` set, job calls need the header `X-Api-Key`.
- Image links are not key-protected (an `<img>` cannot send headers), but job ids are random.

## Stitch from the command line

```bash
python stitch_v4.py capture_data.zip out/     # -> out/mosaic.jpg, out/preview.jpg, out/result.json
python stitch_v4.py walk.mp4 out/             # a plain video works too
```

## What v4 does (vs v3)

- **Input:** uses the app's sharp keyframes directly. Where image matching fails, it uses the phone's own tracking as a weak link, but only if the phone tracked continuously between the two photos.
- **Memory:** video input is streamed; loop search only looks at frames that are part of the mosaic; exposure balancing works per photo footprint instead of on the whole canvas (v3 needed ~11 GB for a 20 m slab).
- **Resolution:** alignment is solved at about 1024 px, then the mosaic is **rendered from the full-resolution photos**. Colours are blended at low resolution; the detail for each pixel comes from the single most central photo. That keeps rebar sharp: averaging several views would smear anything that sticks up from the slab. Output is capped at 80 MP.
- **Scale:** mm per pixel from the app's tape footprint check.
- **Tested on:**
  - the granite walk: 0.79 px median error, 25 loop closures, 100 s
  - phone-resolution 1080p frames: 24.5 MP output in 108 s on 2 CPU cores
  - a forced break: bridged by phone-tracking links

It still assumes the surface is roughly flat. Tall things (walls, stairs, machines) cannot line up.
