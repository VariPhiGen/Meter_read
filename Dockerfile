# ============================================================================
#  Meter OCR pipeline - one Dockerfile, four targets.
#
#    docker build -t meter-ocr .                      # default: both engines, CPU
#    docker build --target slim -t meter-ocr:slim .   # easyocr only, smaller
#    docker build --target gpu  -t meter-ocr:gpu  .   # easyocr on CUDA 12.4
#
#  Run it 24/7, reading every image you drop into .\images :
#
#    docker run -d --name meter-ocr --restart unless-stopped ^
#      -v "%cd%\images:/app/images:ro" ^
#      -v "%cd%\data:/app/data" ^
#      -v "%cd%\logs:/app/logs" ^
#      -v "%cd%\config.yaml:/app/config.yaml:ro" ^
#      meter-ocr
#
#  Everything else is a subcommand on the same image - see README.md.
# ============================================================================


# ---------------------------------------------------------------------------
#  base - system packages and the dependencies every target shares
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS base

# ffmpeg          - RTSP capture goes through cv2.CAP_FFMPEG
# libglib2.0-0    - still required by opencv-python-headless
# ca-certificates - TLS for the one-time model downloads below
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Kolkata

WORKDIR /app


# ---------------------------------------------------------------------------
#  easyocr-core - torch + easyocr with the weights baked in.
#  Shared by the `cpu` and `both` targets so neither pays for it twice.
# ---------------------------------------------------------------------------
FROM base AS easyocr-core

# torch from the CPU index. The default PyPI wheel drags in the entire CUDA
# runtime (~2.5GB) even when it will never touch a GPU.
RUN pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the EasyOCR weights (~100MB) in. Without this, every fresh container
# downloads them on first read - which breaks air-gapped deployment and makes
# the first reading silently slow.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"


# ---------------------------------------------------------------------------
#  gpu - EasyOCR on CUDA 12.4. Needs `--gpus all` at run time AND
#  `gpu: true` in config.yaml; miss either and torch quietly uses the CPU.
#
#  NOTE: this target ships EasyOCR only, so it also needs `engine: easyocr`
#  in config.yaml - which contradicts the measured default. GPU acceleration
#  for paddle needs the paddlepaddle-gpu wheel, which is a different package
#  built against its own CUDA. Add it here if you need paddle on the GPU.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip ffmpeg libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Kolkata

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# gpu=False here only chooses where the weights load for this throwaway call;
# the download path and the files on disk are identical.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY meter_ocr/ ./meter_ocr/
COPY config.yaml ./
RUN mkdir -p images data/snapshots data/annotated logs

ENTRYPOINT ["python", "-m", "meter_ocr"]
CMD ["watch"]


# ---------------------------------------------------------------------------
#  slim - EasyOCR only, ~1.5GB smaller. Build this if you have measured
#  easyocr to be good enough for your meter and want the smaller image.
#  Requires `engine: easyocr` in config.yaml.
# ---------------------------------------------------------------------------
FROM easyocr-core AS slim

COPY meter_ocr/ ./meter_ocr/
COPY config.yaml ./
RUN mkdir -p images data/snapshots data/annotated logs

ENTRYPOINT ["python", "-m", "meter_ocr"]
CMD ["watch"]


# ---------------------------------------------------------------------------
#  DEFAULT TARGET - both engines.
#
#  Last stage in the file, so a bare `docker build .` produces this one. It
#  carries easyocr AND paddleocr because on the meter photos in this project
#  paddleocr is the better of the two (it localises the whole register and
#  reads the "MD" page label at 0.98; easyocr finds only 2-3 characters), and
#  because `compare` needs both present to be able to re-check that on your
#  images. See README.md for the measured numbers.
# ---------------------------------------------------------------------------
FROM easyocr-core AS full

# Pinned as a pair: paddleocr 2.7.x is the last line whose .ocr(img, cls=True)
# call signature matches engines.PaddleOCREngine. 3.x reorganised the API.
RUN pip install --no-cache-dir         paddlepaddle==2.6.2         paddleocr==2.7.3

# Bake the paddle weights in too, for the same reason as easyocr's: no
# first-run download, and the image works with no internet.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', show_log=False)"

# Code last: editing a .py rebuilds only this layer, leaving the torch install
# and both model downloads above cached.
COPY meter_ocr/ ./meter_ocr/
COPY config.yaml ./

# images/ is the drop folder; the rest is runtime output. All are normally
# bind-mounted over, but they must exist for the container to run with no
# mounts at all.
RUN mkdir -p images data/snapshots data/annotated logs

# The entrypoint is the CLI, so the subcommand is all you pass:
#   docker run --rm meter-ocr selftest
#   docker run --rm meter-ocr image images/meter_kwh.jpg
ENTRYPOINT ["python", "-m", "meter_ocr"]
CMD ["watch"]
