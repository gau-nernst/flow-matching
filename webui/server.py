import base64
import gc
import io
import json
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from modelling import MODELS, Flux2Pipeline, ZImagePipeline, load_pipeline

APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"
IMAGE_DIR = APP_ROOT / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DEFAULTS = {name: spec.sample_defaults for name, spec in MODELS.items()}

PIPELINE: Flux2Pipeline | ZImagePipeline | None = None
ACTIVE_MODEL: str | None = None
ATTENTION_IMPL = "pt"
THREAD_POOL = ThreadPoolExecutor(max_workers=1)

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health():
    return None


@app.get("/")
def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/models")
def list_models():
    return {
        "models": list(MODEL_DEFAULTS.keys()),
        "defaults": MODEL_DEFAULTS,
        "active_model": ACTIVE_MODEL,
        "attention_impl": ATTENTION_IMPL,
    }


def _set_attention_impl(attention_impl: str) -> int:
    if PIPELINE is None:
        return 0

    model = PIPELINE.model
    updated = 0
    for module in model.modules():
        if hasattr(module, "attn_impl"):
            module.attn_impl = attention_impl
            updated += 1
    return updated


@app.post("/attention/{attention_impl}")
def set_attention_impl(attention_impl: str):
    global ATTENTION_IMPL
    if attention_impl not in {"pt", "qk-fp8"}:
        raise HTTPException(status_code=400, detail="Unsupported attention implementation")

    ATTENTION_IMPL = attention_impl
    return {"attention_impl": attention_impl, "updated_modules": _set_attention_impl(attention_impl)}


@app.get("/proxy")
def image_proxy(url: str) -> Response:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "application/octet-stream")
    return Response(content=resp.content, media_type=content_type)


@app.get("/image", response_class=JSONResponse)
def image_list():
    items = []
    for path in IMAGE_DIR.iterdir():
        if not (path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}):
            continue
        items.append({"filename": path.name})
    return items


@app.get("/image/{filename}", response_class=FileResponse)
def image_get(filename: str):
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=404, detail="Unknown image")
    path = IMAGE_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown image")
    return FileResponse(path)


@app.post("/image", response_class=JSONResponse)
async def image_import(
    file: UploadFile = File(...),
):
    data = await file.read()
    filename = file.filename or ""
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if "/" in filename or "\\" in filename or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = IMAGE_DIR / filename
    if path.exists():
        raise HTTPException(status_code=409, detail="File already exists")
    Image.open(io.BytesIO(data))
    path.write_bytes(data)


def _cap_pixels(img: Image.Image, count: int):
    w, h = img.size
    if w * h <= count:
        return img

    scale = (count / (w * h)) ** 0.5
    new_w = round(w * scale)
    new_h = round(h * scale)
    return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    neg_prompt: str = Form(""),
    width: int = Form(512),
    height: int = Form(512),
    cfg_scale: float = Form(1.0),
    num_steps: int = Form(4),
    seed: int | None = Form(None),
    images: list[UploadFile] = File(default=[]),
):
    kwargs = dict(
        prompt=prompt,
        neg_prompt=neg_prompt,
        img_size=(height, width),
        cfg_scale=cfg_scale,
        num_steps=num_steps,
        seed=seed,
        pbar=False,
    )

    # Flux2-klein specific logic
    limit_pixels = 2048 * 2048 if len(images) == 1 else 1024 * 1024
    ref_imgs = []
    for image in images:
        data = await image.read()
        img = Image.open(io.BytesIO(data))
        img = _cap_pixels(img, limit_pixels)
        ref_imgs.append(img.convert("RGB"))

    if ref_imgs:
        kwargs.update(ref_imgs=ref_imgs)

    stream_queue: queue.Queue[str | None] = queue.Queue()

    def emit(payload: dict) -> None:
        stream_queue.put(json.dumps(payload, separators=(",", ":")) + "\n")

    def run_job() -> None:
        try:
            # emit progress
            def progress_cb(step: int, total: int) -> None:
                emit({"type": "progress", "step": step, "total": total})

            img_pt = PIPELINE.generate(**kwargs, progress_cb=progress_cb)
            img_np = img_pt.squeeze(0).cpu().numpy()
            pil_img = Image.fromarray(img_np)

            # for the last event, emit the image in base64
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format="WEBP", lossless=True)
            img_bytes.seek(0)
            encoded = base64.b64encode(img_bytes.getvalue()).decode("ascii")
            emit({"type": "done", "image_base64": encoded, "width": width, "height": height})

        except Exception as exc:
            emit({"type": "error", "message": str(exc)})

        finally:
            stream_queue.put(None)  # end signal

    THREAD_POOL.submit(run_job)

    def stream():
        while True:
            item = stream_queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/model/{model_name}")
def load_model(model_name: str):
    global PIPELINE, ACTIVE_MODEL
    if PIPELINE is not None:
        del PIPELINE
        gc.collect()
        torch.cuda.empty_cache()

    PIPELINE = load_pipeline(model_name)
    _set_attention_impl(ATTENTION_IMPL)
    PIPELINE.cuda()
    ACTIVE_MODEL = model_name


@app.get("/vram")
def vram() -> JSONResponse:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
    }
