import asyncio
import io
import os
from pathlib import Path

import uvicorn
from PIL import Image, ImageOps
from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from exhibition_logic import (
    BASE_DIR,
    GENERATED_DIR,
    analyze_cat_image,
    generate_contract,
    parse_date_input,
    query_lucky_day,
)


app = FastAPI(title="纳猫复原互动展项", version="1.0.0")

STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")

MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", "8")) * 1024 * 1024
AUTO_COMPRESS_THRESHOLD = 1 * 1024 * 1024
AUTO_COMPRESS_MAX_DIMENSION = 1600


def _compress_image_if_needed(image_bytes: bytes, filename: str) -> tuple[bytes, str]:
    if len(image_bytes) <= AUTO_COMPRESS_THRESHOLD:
        return image_bytes, filename

    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)

    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A") if "A" in image.getbands() else None
        background.paste(image.convert("RGBA"), mask=alpha)
        image = background
    else:
        image = image.convert("RGB")

    max_side = max(image.size)
    if max_side > AUTO_COMPRESS_MAX_DIMENSION:
        scale = AUTO_COMPRESS_MAX_DIMENSION / max_side
        resized_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(resized_size, Image.Resampling.LANCZOS)

    quality_candidates = [88, 82, 76, 70, 64, 58, 52, 46, 40]
    best_bytes = image_bytes
    for quality in quality_candidates:
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        candidate = output.getvalue()
        if len(candidate) < len(best_bytes):
            best_bytes = candidate
        if len(candidate) <= AUTO_COMPRESS_THRESHOLD:
            return candidate, f"{Path(filename).stem}.jpg"

    return best_bytes, f"{Path(filename).stem}.jpg"


async def read_image_upload(photo: UploadFile) -> tuple[bytes, str]:
    if not photo.filename:
        raise HTTPException(status_code=400, detail="请上传图片文件")
    if not (photo.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持上传图片文件")

    image_bytes = await photo.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="请上传猫咪照片")
    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"图片不能超过 {MAX_UPLOAD_SIZE // 1024 // 1024}MB")
    compressed_bytes, normalized_filename = _compress_image_if_needed(image_bytes, photo.filename)
    return compressed_bytes, normalized_filename


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/cat/read")
async def cat_read(
    photo: UploadFile = File(...),
    nickname: str = Form(default=""),
) -> JSONResponse:
    image_bytes, normalized_filename = await read_image_upload(photo)

    try:
        result = await run_in_threadpool(
            analyze_cat_image,
            image_bytes,
            normalized_filename,
            nickname.strip() or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"猫咪图片解析失败：{exc}") from exc
    return JSONResponse(result)


@app.post("/api/date/lucky")
async def lucky_day(query_date: str = Form(default="")) -> JSONResponse:
    try:
        target_date = parse_date_input(query_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="日期格式错误，请使用 YYYY-MM-DD") from exc
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(query_lucky_day, target_date),
            timeout=50.0,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="黄历查询调用大模型超时：后端在 50 秒内未得到模型结果，请检查模型响应速度、网络状态，或切换更快模型。",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    return JSONResponse(result)


@app.post("/api/contract/generate")
async def contract_generate(
    photo: UploadFile = File(...),
    event_text: str = Form(...),
    owner_name: str = Form(default=""),
    cat_name: str = Form(default=""),
    contract_date: str = Form(default=""),
) -> JSONResponse:
    image_bytes, normalized_filename = await read_image_upload(photo)
    if not event_text.strip():
        raise HTTPException(status_code=400, detail="请填写纳猫事件")
    if len(event_text.strip()) > 200:
        raise HTTPException(status_code=400, detail="纳猫事件请控制在 200 字以内")

    try:
        parsed_date = parse_date_input(contract_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="立契日期格式错误，请使用 YYYY-MM-DD") from exc
    try:
        result = await run_in_threadpool(
            generate_contract,
            event_text.strip(),
            owner_name.strip(),
            cat_name.strip(),
            parsed_date,
            image_bytes,
            normalized_filename,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"纳猫契生成失败：{exc}") from exc
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
