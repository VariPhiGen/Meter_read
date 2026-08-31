# Meter OCR Pipeline

Reads the digits off an electricity meter and appends them to a CSV.

Two input modes, one image:

- **watch** (default) — runs 24/7, OCRs every image you drop into `images\`
- **schedule** — hourly readings from RTSP cameras, for when the cameras arrive

Everything runs in Docker. There is nothing to install on the host but Docker
itself, and one Dockerfile builds every variant.

---

## 1. Install Docker (not yet installed on this machine)

Docker Desktop needs the WSL2 backend. In an **Administrator** PowerShell:

```powershell
wsl --install
```

Reboot, install Docker Desktop from
<https://docs.docker.com/desktop/install/windows-install/>, then confirm:

```powershell
docker --version
```

## 2. Build

```powershell
docker build -t meter-ocr .
```

First build takes a while — it pulls PyTorch and bakes **both** engines'
weights into the image, so containers start instantly afterwards and need no
internet access at run time.

Optional variants, same file:

```powershell
docker build --target slim -t meter-ocr:slim .   # EasyOCR only, ~1.5GB smaller
docker build --target gpu  -t meter-ocr:gpu  .   # CUDA 12.4 (EasyOCR only)
```

## 3. Check it works

```powershell
docker run --rm meter-ocr selftest
```

Generates a synthetic meter panel, runs it through the real OCR path, and
reports pass/fail per stage. No camera, no photo, no network needed.

## 4. Run it 24/7

```powershell
docker run -d --name meter-ocr --restart unless-stopped ^
  -v "%cd%\images:/app/images:ro" ^
  -v "%cd%\data:/app/data" ^
  -v "%cd%\logs:/app/logs" ^
  -v "%cd%\config.yaml:/app/config.yaml:ro" ^
  meter-ocr
```

That is the whole thing. It now sits there watching `images\`. Drop a JPG in
and a row appears in `data\readings.csv` within a few seconds.

```powershell
docker logs -f meter-ocr        # watch it work
docker restart meter-ocr        # apply a config.yaml change
docker stop meter-ocr           # stop
```

`--restart unless-stopped` brings it back after a reboot or a crash, which is
what makes it genuinely 24/7.

Your input images are **never modified, moved or deleted**. The container
tracks what it has already read in `data\.watch_state.json`; re-saving an image
changes its timestamp and it gets read again, which is what you want when
re-testing a tweaked crop.

---

## Reading images: the accuracy loop

Drop images in, then look at `data\annotated\<name>_annotated.jpg`. It shows the
crop exactly as OCR saw it, with every detection boxed and labelled. That
picture tells you what to fix.

**The ROI is the single biggest accuracy setting.** Reading a whole meter gives
you the reading, the serial number and the voltage rating concatenated:

```
without ROI :  08726461.447198224015052     <- garbage
with ROI    :  08726.4                      <- correct
```

`watch.roi` in `config.yaml` takes three kinds of value:

| Value | Use when |
|---|---|
| `auto` | **default** — handheld photos. Finds the lit LCD panel in each image by its green backlight and crops to it, so every shot can be framed differently. |
| `[x, y, w, h]` | a mounted camera that frames the meter identically every time. |
| `null` | the photo is already cropped tight to the digits. |

`auto` pads the detected box outwards, because the outermost digit usually sits
right at the edge of the backlight — without padding a leading `0` gets clipped
and `02.32` silently becomes `2.32`.

For a fixed box, get the numbers by opening the photo in any image viewer that
shows pixel coordinates (the Paint status bar does), or try one without
rebuilding:

```powershell
docker run --rm -v "%cd%\images:/app/images:ro" -v "%cd%\data:/app/data" ^
  meter-ocr image images/meter_kwh.jpg --roi auto
```

### Panel-only reading

The green glass carries more than the number: `kWh`, `kW`, `MD`, and a subscript
decimal group. Two settings keep the reading and drop the rest.

**`ocr.min_height_frac: 0.55`** — keep only text at least 55% as tall as the
tallest thing on the panel. The main register is about twice the height of the
labels. This matters more than it sounds: with a digits-only allowlist the
recogniser is *forced* to return a digit for `kW`, and it comes back as `6`
scored 0.98 — higher than the real digits. Confidence cannot separate them.
Height can. Set to `0` for a plain printed meter with no labels.

**`ocr.target_width: 1600`** — resize every crop to about this width, up *or*
down. A panel crop out of an 8000px phone photo is ~4500px wide; multiplying
that by `upscale: 3.0` produces a 13779px image that is slower by orders of
magnitude and no more readable. A 200px crop off a low-res camera genuinely
needs enlarging. Normalising both to one width lets one set of preprocessing
numbers serve both.

Then, in order:

1. **ROI** — as above. Do this first; nothing else comes close.
2. **Focus** — no software fixes an out-of-focus photo.
3. **Lighting** — glare on the meter glass is the most common cause of misreads.
   Shoot slightly off-perpendicular so reflections miss the lens.
4. `invert: true` — if the meter is light digits on a dark background.
5. `threshold: true` — if the image is noisy.
6. `decimals: 2` — if the meter always has 2 decimal places. Stops a smudge
   being read as a decimal point and shifting the value by 100x.
7. `upscale: 4.0` — if the digits are small in frame.

---

## Which OCR engine? — measured, 2026-08-29

Both were run on this project's own photos, on byte-identical crops through
identical preprocessing, so the only variable was the recogniser:

| image | truth | easyocr | paddleocr |
|---|---|---|---|
| `meter_kw.jpg` | `02.32` | `06` (0.91) | *nothing detected* |
| `meter_kwh.jpg` | `13200` | `288` (0.70) | `1328883` (0.85) |
| **score** | | **0/2** | **0/2** |

**Neither reads this meter correctly yet.** But they fail differently, and the
difference decides which is worth building on:

- **EasyOCR finds only 2-3 of the 5-6 digits.** Asked for all text with no
  allowlist it returned `'p:'`, `'kli'`, `'kM'`. It is not locating the glyphs.
- **PaddleOCR localises the whole register** (`1328883` — seven characters for
  a six-character display) and read the `MD` page label at **0.98**. Its errors
  are confined to one systematic confusion: `0` read as `8`.

That confusion is the meter's fault, not the engine's. On this LCD the *unlit*
segments are still faintly visible, so every `0` carries a ghost `8` behind it.
Thresholding cannot separate them: at a 5% ink cut the lit segments fragment
(glare makes them non-uniform), at 15% the ghosts fill in solid.

**`paddleocr` is therefore the configured default** — it is the better
foundation, being wrong only in a specific, characterisable way.

### Re-running this on your own images

```powershell
docker run --rm -v "%cd%\images:/app/images:ro" -v "%cd%\data:/app/data" ^
  -v "%cd%\truth.csv:/app/truth.csv:ro" ^
  meter-ocr compare --truth truth.csv
```

Both engines are in the default image, so no rebuild is needed. Put the correct
readings in `truth.csv` first; results are written to
`data\engine_comparison.csv`.

### Getting to a correct reading

In order of effort:

1. **Better photographs — by far the cheapest win.** These were shot through
   glare, with the panel filling ~10% of the frame. Fill the frame with the
   display, light it evenly and diffusely, and shoot slightly off-perpendicular
   so reflections miss the lens. Much of the ghosting is glare washing out the
   contrast between lit and unlit segments, and that is fixable at the camera.
2. **A seven-segment reader.** Locate each digit cell and measure the darkness
   of its seven segment regions *relative to the other cells in the same
   image*. Ghost segments are consistently lighter than lit ones within a
   frame, so a relative test separates them where a global threshold cannot.
   This is deterministic and needs no training data.
3. **A small digit classifier** trained on your own snapshots. The pipeline
   saves every image it reads, so the training data accumulates from day one.

`engines.py` puts every engine behind one interface, so options 2 and 3 slot in
beside the two already there without touching the rest of the pipeline.

---

## Multi-page meters

This meter cycles between an energy page (`kWh`) and a maximum-demand page
(`kW` / `MD`). Logging both into one column is meaningless — the kW page's
`02.32` next to the kWh page's `13200` reads like the meter reset.

```yaml
watch:
  page: "kWh"              # record only this page; null = record every page
  read_unknown_page: true  # if the label is unreadable, read anyway
```

The page is identified from the unit label, read without the digits-only
allowlist so letters can come back. On this meter the `MD` label is the
reliable marker, so a kW-page photo is skipped:

```
  image      meter_kw.jpg
  status     SKIPPED  (MD page, wanted kWh)
```

`read_unknown_page: true` means an unreadable label produces a reading rather
than a silent gap. Set it `false` to skip whenever unsure.

---

## RTSP cameras

Fill in the `cameras:` list in `config.yaml`:

```yaml
cameras:
  - id: "meter_01"
    name: "Meter 1"
    enabled: true
    url: "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"
    roi: [420, 300, 260, 90]
    rotate: 0
    decimals: 2
```

| Brand           | URL |
|-----------------|-----|
| Hikvision       | `rtsp://user:pass@IP:554/Streaming/Channels/101` |
| Dahua / CP Plus | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| Axis            | `rtsp://user:pass@IP/axis-media/media.amp` |
| Generic ONVIF   | `rtsp://user:pass@IP:554/onvif1` |

Use the **main stream**, not the sub-stream — the sub-stream is usually too
low-resolution to read digits.

```powershell
docker run --rm -v "%cd%\config.yaml:/app/config.yaml:ro" -v "%cd%\data:/app/data" meter-ocr cameras
docker run --rm -v "%cd%\config.yaml:/app/config.yaml:ro" -v "%cd%\data:/app/data" meter-ocr once
```

Then switch the long-running container from `watch` to `schedule` by appending
that subcommand to the `docker run` line in step 4.

---

## Commands

Every mode is a subcommand on the same image:

| Command | What it does |
|---|---|
| `watch` | **default** — OCR every image dropped into the watch folder, forever |
| `image FILE` | read one image and print the result |
| `compare` | score OCR engines against `truth.csv` |
| `once` | one RTSP reading from every enabled camera |
| `schedule` | hourly RTSP loop, forever |
| `selftest` | synthetic end-to-end check, no camera needed |
| `cameras` | camera reachability, focus score, and what OCR reads now |

Global flags: `--engine easyocr|paddleocr`, `--cpu`, `--config PATH`.

```powershell
docker run --rm meter-ocr --help
```

## Output

`data\readings.csv`:

```csv
timestamp_local,timestamp_utc,source,camera_id,camera_name,reading,raw_text,confidence,votes,frames,delta,status,note,snapshot
2026-08-29 14:00:02,2026-08-29 08:30:02,meter_kwh.jpg,image,image,13200.0,13200,0.940,3/3,3,1.240,OK,,data/annotated/meter_kwh_annotated.jpg
```

- **votes** — how many reads agreed. For a still image the votes come from
  three preprocessing variants (as configured, inverted, thresholded); for a
  camera they come from five consecutive frames. `3/3` is solid, `1/3` is shaky.
- **status** — `OK`, `SUSPECT` (read succeeded but failed a sanity check, e.g.
  the reading went backwards) or `FAILED` (nothing readable).

Every attempt is logged whatever happens — an unreadable image produces a
`FAILED` row, not a gap.

## Layout

```
Dockerfile          every image target: full (default), slim, gpu
config.yaml         all settings - the only file you edit
requirements.txt    pinned dependencies
truth.csv           known-correct readings, for `compare`
images/             drop images here (mounted into the container)
data/               readings.csv, annotated/, snapshots/
logs/               rotating logs
meter_ocr/
  __main__.py       the CLI - every subcommand above
  config.py         config.yaml -> dataclasses
  engines.py        EasyOCR / PaddleOCR behind one interface
  ocr.py            cleaning, parsing, majority voting
  preprocess.py     ROI crop, contrast, sharpening, variants
  autocrop.py       finds the lit LCD panel, for roi: auto
  pages.py          which page the meter is showing (kWh / kW / MD)
  images.py         read a still image; write the annotated copy
  watch.py          the 24/7 folder watcher
  capture.py        RTSP frame grabbing (TCP, buffer flush, retries)
  pipeline.py       one full camera cycle
  schedule.py       the hourly loop
  compare.py        engine scoring
  storage.py        CSV, snapshots, validation
  cameras.py        connectivity check
  selftest.py       synthetic end-to-end check
```

## Things that will bite you

**Timezone.** The scheduler fires on local wall-clock time. Containers default
to UTC, so without `TZ` your "hourly at :00" runs on the wrong hour. It is set
to `Asia/Kolkata` in the Dockerfile — change it there, or pass `-e TZ=...`.

**numpy must stay below 2.0.** easyocr 1.7.2 and the cu124 torch build are
compiled against the numpy 1.x ABI. numpy 2.x makes them fail at import with a
binary-incompatibility error. `requirements.txt` pins `1.24.4`.

**GPU is opt-in three times.** Build `--target gpu`, pass `--gpus all`, and set
`gpu: true` in `config.yaml`. Miss any one and torch silently falls back to the
CPU (`engines.py` logs a warning when `gpu: true` but CUDA is unavailable).

**USB webcams do not pass through** to containers on Docker Desktop for Windows.
This project uses RTSP URLs and dropped-in images, so it is unaffected.

**A mounted file wins over the baked one.** `config.yaml` is inside the image,
but the run command mounts the host copy over it so you can edit settings and
just `docker restart`. Drop that `-v` line if you would rather ship one frozen
artefact with its config sealed in.

**This folder is inside OneDrive.** `data\` grows steadily as snapshots and
annotated copies accumulate. Right-click it in File Explorer → *Always keep on
this device*, or exclude it in OneDrive Settings → Account → *Choose folders*.
