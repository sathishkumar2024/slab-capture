# Slab Capture — guided phone capture for floor/slab stitching tests

A web app that runs in the phone browser (Android Chrome, iPhone Safari). No install and no app store.
While you walk it shows live guidance: phone level, walking speed, blur, low texture, glare, turning speed and a coverage map.
It records:

- **video** (MediaRecorder, 12 Mbit/s by default)
- **sharp keyframe photos** (JPEG straight from the camera stream, picked automatically every ~15% of frame motion)
- **sensor logs**: gyro, accelerometer and orientation, plus per-frame tracking, tilt and sharpness, all on one clock

Everything stays on the phone (browser storage, survives a crash or reload) until you export it.

## 1. Put it online (one time, ~3 minutes)

The camera only works on an `https://` page, so the files need hosting. GitHub Pages is free:

1. On github.com create a new **public** repository, e.g. `slab-capture`.
2. **Add file → Upload files**. Upload `index.html`, `manifest.webmanifest`, `sw.js`, `icon.svg`, then **Commit**.
3. **Settings → Pages**. Under "Build and deployment" choose **Deploy from a branch → main → / (root) → Save**.
4. After about a minute the app is live at `https://<your-username>.github.io/slab-capture/`.

Any static HTTPS host works just as well, e.g. Netlify Drop (drag the folder onto app.netlify.com/drop).

## 2. On the phone

- Open the link **once with internet before going to site**; after that it also opens offline.
- Allow **camera**. On iPhone, also allow **Motion & Orientation**; without it there is no level bubble and no gyro.
- Optional: Share → **Add to Home Screen** so it opens like an app.
- **Android:** keep "Full screen + lock landscape" ticked.
- **iPhone:** a web page can't lock the screen orientation. Turn **rotation lock OFF**, hold the phone **sideways**, then point it down. If the picture flips mid-recording the screen border flashes red; stop and restart that pass.
- Don't clear browser data until you have exported your sessions.

### Reading the screen

| Element | Meaning |
|---|---|
| Top banner | What to fix right now. Red = fix immediately, amber = adjust, green = good, blue = paused (good moment to turn). |
| Level bubble | Keep it in the green centre (< 8° tilt). |
| Speed bar | Keep the marker in the green. The unit is "screen heights per second", so it works at any height. |
| Coverage map (bottom right) | Where you have been. Red = seen once, green = seen 5+ times. It is approximate and drifts a little on long walks. |
| Top right | Recording time, photos saved, storage used, resolution and fps. |

The gyro calibrates itself during the first turn or two. The map's heading gets noticeably more accurate after that.

## 3. Site plan for tomorrow

Aim for 5–6 short sessions rather than one long one; 1–3 minutes each is ideal.

**Before each session (1 minute):**

1. Lay a tape measure flat on the surface.
2. Hold the phone at your capture height, pointing straight down.
3. Read how many metres of tape span the **short side** of the preview.
4. Enter that in "Footprint check". The app then shows mm/pixel and how far apart to walk the lanes.

**Sessions, in priority order:**

1. **Rebar mat, 3-lane raster.** Lanes ~5 m long, about 50% overlap between lanes. Leave the tape measure in the area (ground truth). Finish by walking back over the start.
2. **Same area with the phone's normal camera app.** Walk the same raster. This compares native-camera video (stabilised, better encoder) against the browser capture.
3. **Plain concrete slab**, 2 lanes. This is the low-texture test.
4. **Rebar at a different height.** Arm raised (~1.6 m), or a pole if you have one. Enter the new height and footprint.
5. **Rebar at a normal walking pace**, ignoring the speed warnings. This is a stress test to see where things break.
6. If there's time: **mixed sun and shadow**, and a pass with workers moving through.

**Write down for each session** (the Notes field works): rebar height above the deck, bar diameter and spacing (measure 2–3 with the tape). These are the ground truth for later spacing and diameter checks.

**After each session:**

1. Tap **STOP** (the same button as REC). The summary screen opens.
2. Save both files:
   - **iPhone:** tap **Share…** → **Save to Files**. Or send them by AirDrop or WhatsApp, which gives you both files together.
   - **Android:** tap **Save video** and **Save data (.zip)**. They go to Downloads.
   - If a button turns **green ("Tap to save ✓")**, tap it once more. The phone only allows saving right after a tap, and preparing a large file can take longer than that.
3. If storage runs low, delete exported sessions from the list.
4. At the end of the day, copy everything to your laptop.

**Storage:** at 1080p expect roughly 200 MB per minute (video plus photos). A red **STORAGE FULL** banner means stop and export.

## 4. What's in the export

`<name>_<id>.mp4` (or `.webm` on some phones), plus `<name>_<id>_data.zip` containing:

| File | Contents |
|---|---|
| `session.json` | notes, device, camera settings/capabilities, quality summary, gyro calibration factor |
| `keyframes/kf_000123.jpg` + `keyframes.csv` | sharp photos with timestamp, tracker pose, tilt, sharpness |
| `frames.csv` | every processed frame: tracking, pose, speed, tilt, sharpness, brightness, guidance status |
| `imu.csv`, `orientation.csv` | raw `devicemotion` / `deviceorientation` events |
| `events.csv` | record start/stop, warnings, gyro calibration, orientation changes |
| `coverage.png` | the coverage map |

All logs share `t_ms` = milliseconds since REC was pressed. Poses are the app's live tracker (units: the frame's short side = 1). Treat them as an initial guess for the stitcher, not ground truth.

```
python tools/load_session.py capture_data.zip                         # quality report
python tools/load_session.py capture_data.zip --extract out/          # unzip photos + logs
python tools/load_session.py capture_data.zip --video capture.mp4 --stitch mosaic.jpg   # needs stitch_v3.py next to it
```

**If something goes wrong,** a red or amber box appears with the exact error. Screenshot it and send it.
An amber *"Phone storage is not available"* box means recording still works, but it's kept in memory only. Don't close or reload the page until you've exported.

## 5. Known limits (v0.1)

- **Browser video is not the phone's best camera pipeline.** There's no manual exposure or focus lock, and the encoder is chosen by the browser. That's why session 2 compares against the native camera.
- **The map drifts over long walks.** Tall things in view (railings, columns, feet) bias it; the gyro fixes most of this once calibrated.
- **iPhone:** no vibration, no orientation lock (see above).
- **It doesn't measure height.** The "Footprint check" gives scale; markers or ARCore/ARKit come later in the native app.
