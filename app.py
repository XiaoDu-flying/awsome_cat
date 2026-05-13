import os
from pathlib import Path

import uvicorn
from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from exhibition_logic import (
    BASE_DIR,
    GENERATED_DIR,
    UPLOAD_DIR,
    analyze_cat_image,
    generate_contract,
    parse_date_input,
    query_lucky_day,
    save_bytes_file,
)


app = FastAPI(title="纳猫复原互动展项", version="1.0.0")

STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")

MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", "8")) * 1024 * 1024


async def read_image_upload(photo: UploadFile) -> bytes:
    if not photo.filename:
        raise HTTPException(status_code=400, detail="请上传图片文件")
    if not (photo.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持上传图片文件")

    image_bytes = await photo.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="请上传猫咪照片")
    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"图片不能超过 {MAX_UPLOAD_SIZE // 1024 // 1024}MB")
    return image_bytes


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
    image_bytes = await read_image_upload(photo)

    save_bytes_file(image_bytes, Path(photo.filename or "cat.png").suffix or ".png", UPLOAD_DIR, "cat")
    try:
        result = await run_in_threadpool(
            analyze_cat_image,
            image_bytes,
            photo.filename or "cat.png",
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
    return JSONResponse(await run_in_threadpool(query_lucky_day, target_date))


@app.post("/api/contract/generate")
async def contract_generate(
    photo: UploadFile = File(...),
    event_text: str = Form(...),
    owner_name: str = Form(default=""),
    cat_name: str = Form(default=""),
    contract_date: str = Form(default=""),
) -> JSONResponse:
    image_bytes = await read_image_upload(photo)
    if not event_text.strip():
        raise HTTPException(status_code=400, detail="请填写纳猫事件")
    if len(event_text.strip()) > 200:
        raise HTTPException(status_code=400, detail="纳猫事件请控制在 200 字以内")

    try:
        parsed_date = parse_date_input(contract_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="立契日期格式错误，请使用 YYYY-MM-DD") from exc
    save_bytes_file(image_bytes, Path(photo.filename or "contract.png").suffix or ".png", UPLOAD_DIR, "contract")
    try:
        result = await run_in_threadpool(
            generate_contract,
            event_text.strip(),
            owner_name.strip(),
            cat_name.strip(),
            parsed_date,
            image_bytes,
            photo.filename or "contract.png",
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
