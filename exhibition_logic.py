import hashlib
import io
import math
import os
import textwrap
import uuid
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from llm_api import chat_json, chat_text, get_last_llm_diagnostic, get_llm_status


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
GENERATED_DIR = BASE_DIR / "generated"

UPLOAD_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)


def _llm_enrichment_enabled() -> bool:
    configured = os.getenv("ENABLE_LLM_ENRICHMENT")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    return any(
        os.getenv(name)
        for name in ["ARK_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"]
    )


def save_bytes_file(data: bytes, suffix: str, folder: Path, prefix: str) -> Path:
    file_path = folder / f"{prefix}_{uuid.uuid4().hex}{suffix}"
    file_path.write_bytes(data)
    return file_path


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    if bold:
        candidates = [
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        ] + candidates
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _image_stats(image_bytes: bytes) -> dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    small = image.resize((64, 64))
    pixels = list(small.getdata())
    total = len(pixels)
    avg_r = sum(pixel[0] for pixel in pixels) / total
    avg_g = sum(pixel[1] for pixel in pixels) / total
    avg_b = sum(pixel[2] for pixel in pixels) / total
    brightness = (avg_r * 0.299 + avg_g * 0.587 + avg_b * 0.114) / 255
    color_range = (
        sum(abs(pixel[0] - avg_r) + abs(pixel[1] - avg_g) + abs(pixel[2] - avg_b) for pixel in pixels)
        / total
        / 255
    )
    width, height = image.size
    digest = hashlib.sha256(image_bytes).hexdigest()
    return {
        "width": width,
        "height": height,
        "avg_r": avg_r,
        "avg_g": avg_g,
        "avg_b": avg_b,
        "brightness": brightness,
        "color_range": color_range,
        "digest": digest,
    }


def _dominant_style(stats: dict[str, Any]) -> str:
    r, g, b = stats["avg_r"], stats["avg_g"], stats["avg_b"]
    if r > 165 and g > 140 and b < 130:
        return "金橘"
    if r > 165 and g > 165 and b > 165:
        return "雪团"
    if r < 95 and g < 95 and b < 95:
        return "玄墨"
    if abs(r - g) < 15 and abs(g - b) < 15:
        return "烟灰"
    if r > g and r > b:
        return "花狸"
    return "锦斑"


def _local_cat_result(stats: dict[str, Any], nickname: Optional[str] = None) -> dict[str, Any]:
    style = _dominant_style(stats)
    brightness = stats["brightness"]
    color_range = stats["color_range"]
    digest_int = int(stats["digest"][:8], 16)

    body = "身形圆润" if brightness > 0.6 else "骨相清秀"
    temper = "神气昂然" if color_range > 0.42 else "气质安稳"
    aura = ["颇具福相", "灵气十足", "宜亲宜伴", "颇有主见"][digest_int % 4]

    fortune_score = round(70 + ((brightness * 15) + (color_range * 20) + (digest_int % 12)) / 2)
    legacy_alignment = "好猫" if fortune_score >= 82 else "坏猫"
    label = "福缘好猫" if legacy_alignment == "好猫" else "傲娇奇猫"
    if nickname:
        title = f"{nickname}·{label}"
    else:
        title = label

    return {
        "title": title,
        "appearance": [f"毛色偏{style}", body, temper],
        "legacy_alignment": legacy_alignment,
        "summary": f"此猫观之{aura}，{body}而{temper}，纳之入宅，多半能添几分生趣。",
        "tags": [style, aura.replace("，", ""), "展厅推荐"],
        "fortune_score": min(fortune_score, 99),
    }


def analyze_cat_image(image_bytes: bytes, filename: str, nickname: Optional[str] = None) -> dict[str, Any]:
    stats = _image_stats(image_bytes)
    local_result = _local_cat_result(stats, nickname)

    if not _llm_enrichment_enabled():
        return local_result

    prompt = textwrap.dedent(
        f"""
        请根据猫咪图片与补充信息，输出一个 JSON 对象，不要输出额外说明。
        字段包括：
        - title: 趣味标题
        - appearance: 3条外貌特点数组
        - legacy_alignment: 只能是“好猫”或“坏猫”
        - summary: 60字以内古风点评
        - tags: 3个标签数组
        - fortune_score: 0到100的整数

        补充信息：
        - 猫咪昵称：{nickname or '未提供'}
        - 图片文件名：{filename}
        - 本地视觉特征参考：{local_result}

        要求：
        1. 内容需温和有趣，避免攻击性；
        2. 即便输出“坏猫”，也应体现为调皮、傲娇，而不是负面辱骂；
        3. appearance 和 tags 都必须是数组。
        """
    ).strip()

    llm_result = chat_json(
        prompt,
        system_prompt="你是展馆里的古风猫相师，擅长根据图片生成结构化、友好的相猫结果。",
        image_bytes=image_bytes,
        image_mime_type=_guess_mime_type(filename),
        max_tokens=1200,
    )

    if isinstance(llm_result, dict):
        return {
            "title": str(llm_result.get("title") or local_result["title"]),
            "appearance": _normalize_array(llm_result.get("appearance"), local_result["appearance"]),
            "legacy_alignment": "好猫" if llm_result.get("legacy_alignment") == "好猫" else local_result["legacy_alignment"],
            "summary": str(llm_result.get("summary") or local_result["summary"]),
            "tags": _normalize_array(llm_result.get("tags"), local_result["tags"]),
            "fortune_score": _normalize_score(llm_result.get("fortune_score"), local_result["fortune_score"]),
        }
    return local_result


def _guess_mime_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _normalize_array(value: Any, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        if normalized:
            return normalized[:5]
    return fallback


def _normalize_score(value: Any, fallback: int) -> int:
    try:
        return max(0, min(100, int(value)))
    except Exception:
        return fallback


def _pick_unique_items(pool: list[str], seed: int, step: int, count: int) -> list[str]:
    results: list[str] = []
    index = seed % len(pool)
    while len(results) < count:
        item = pool[index % len(pool)]
        if item not in results:
            results.append(item)
        index += step
    return results


def query_lucky_day(target_date: date) -> dict[str, Any]:
    seed = target_date.toordinal()
    score = (seed * 37 + target_date.day * 9 + target_date.month * 13) % 100
    is_lucky = score >= 45

    if score >= 80:
        level = "大吉"
    elif score >= 60:
        level = "中吉"
    elif score >= 45:
        level = "小吉"
    else:
        level = "平"

    yi_bank = ["备猫粮", "置软垫", "轻声唤名", "整理猫窝", "拍合照", "系祈福牌"]
    ji_bank = ["骤然惊扰", "频繁换粮", "追逐喧闹", "空手抱猫", "忘备饮水", "过度围观"]

    yi = _pick_unique_items(yi_bank, seed, 3, 3)
    ji = _pick_unique_items(ji_bank, seed, 5, 3)
    summary = (
        f"{target_date.month}月{target_date.day}日猫缘{level}，"
        + ("适宜迎猫入宅、缓缓亲近。" if is_lucky else "宜先整备猫居，改日再迎更稳妥。")
    )

    llm_text = None
    if _llm_enrichment_enabled():
        llm_text = chat_text(
            textwrap.dedent(
                f"""
                日期：{target_date.isoformat()}
                基础判定：{summary}
                宜：{', '.join(yi)}
                忌：{', '.join(ji)}
                请写一段不超过60字的古风黄历文案。
                """
            ).strip(),
            system_prompt="你是展馆中的黄历讲解员，请用简洁、古风、友好的中文写作。",
            max_tokens=200,
        )

    return {
        "date": target_date.isoformat(),
        "is_lucky": is_lucky,
        "level": level,
        "score": score,
        "yi": yi,
        "ji": ji,
        "summary": summary,
        "oracle": llm_text or _fallback_oracle(target_date, level, is_lucky),
    }


def _fallback_oracle(target_date: date, level: str, is_lucky: bool) -> str:
    if is_lucky:
        return f"是日猫缘{level}，若备清水软席，再以和声迎之，主相处和顺。"
    return f"是日猫缘{level}，宜先静室净具，暂缓纳猫，改择清朗之期更佳。"


def generate_contract(
    event_text: str,
    owner_name: str,
    cat_name: str,
    contract_date: date,
    image_bytes: bytes,
    source_filename: str,
) -> dict[str, Any]:
    llm_available, llm_status_message = get_llm_status()
    llm_runtime_message = llm_status_message
    base_title = f"纳猫契·{cat_name or '灵猫'}"
    default_body = (
        f"今有{owner_name or '纳猫人'}，以诚心迎{cat_name or '灵猫'}入宅。"
        f"缘起“{event_text}”，自{contract_date.isoformat()}日起，"
        "当以清水、暖食、柔言相待，共守朝夕，同享安然。"
    )
    body = default_body
    ai_generated = False
    if _llm_enrichment_enabled():
        llm_body = chat_text(
                textwrap.dedent(
                    f"""
                    请为展馆互动装置写一段 80 字以内的《纳猫契》正文，文风古雅但易懂。
                    纳猫事件：{event_text}
                    主人称呼：{owner_name or '纳猫人'}
                    猫咪名字：{cat_name or '灵猫'}
                    日期：{contract_date.isoformat()}
                    """
                ).strip(),
                system_prompt="你是文博展项文案师，负责撰写简洁优雅的中文契书。",
                max_tokens=250,
            )
        if llm_body and llm_body.strip():
            body = llm_body.strip()
            ai_generated = True
            llm_runtime_message = "AI 文案生成成功。"
        else:
            llm_runtime_message = get_last_llm_diagnostic() or "已检测到 LLM 配置，但本次正文生成失败，已回退模板。"

    output_filename = f"contract_{uuid.uuid4().hex}.png"
    output_path = GENERATED_DIR / output_filename
    _render_contract_poster(
        image_bytes=image_bytes,
        source_filename=source_filename,
        title=base_title,
        body=body,
        owner_name=owner_name or "纳猫人",
        cat_name=cat_name or "灵猫",
        contract_date=contract_date,
        output_path=output_path,
    )
    return {
        "title": base_title,
        "body": body,
        "ai_generated": ai_generated,
        "body_source": "ai" if ai_generated else "template",
        "body_source_label": "本契正文由 AI 生成" if ai_generated else "本契正文由模板生成",
        "body_source_reason": llm_runtime_message,
        "llm_available": llm_available,
        "image_url": f"/generated/{output_filename}",
        "download_name": output_filename,
    }


def _render_contract_poster(
    *,
    image_bytes: bytes,
    source_filename: str,
    title: str,
    body: str,
    owner_name: str,
    cat_name: str,
    contract_date: date,
    output_path: Path,
) -> None:
    width, height = 1200, 1800
    canvas = Image.new("RGB", (width, height), "#f6ecd2")
    draw = ImageDraw.Draw(canvas)

    for index in range(12):
        alpha_color = 248 - index * 4
        draw.rounded_rectangle(
            [50 + index * 3, 50 + index * 3, width - 50 - index * 3, height - 50 - index * 3],
            radius=28,
            outline=(alpha_color, 180, 120),
            width=2,
        )

    title_font = _font(64, bold=True)
    sub_font = _font(28)
    body_font = _font(34)
    stamp_font = _font(42, bold=True)

    draw.text((width / 2, 140), title, font=title_font, fill="#6a2418", anchor="mm")
    draw.text((width / 2, 210), "第二展厅·纳猫复原互动凭契", font=sub_font, fill="#8b5a2b", anchor="mm")

    cat_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cat_img = _crop_center(cat_img, (520, 520)).filter(ImageFilter.SMOOTH_MORE)
    photo_box = (340, 280, 860, 800)
    canvas.paste(cat_img, photo_box)
    draw.rounded_rectangle(photo_box, radius=26, outline="#8b5a2b", width=4)

    body_block = textwrap.fill(body, width=18)
    draw.multiline_text((140, 900), body_block, font=body_font, fill="#3a2a18", spacing=18)

    details = [
        f"纳猫人：{owner_name}",
        f"猫名：{cat_name}",
        f"立契日：{contract_date.isoformat()}",
        f"凭图源：{Path(source_filename).name}",
    ]
    details_text = "\n".join(details)
    draw.multiline_text((140, 1370), details_text, font=sub_font, fill="#614632", spacing=16)

    draw.ellipse((860, 1380, 1090, 1610), outline="#9f1f1f", width=8)
    draw.text((975, 1495), "纳猫\n有契", font=stamp_font, fill="#9f1f1f", anchor="mm", align="center", spacing=8)

    footer = "愿人猫同居，朝夕有伴，安暖如常。"
    draw.text((width / 2, 1710), footer, font=sub_font, fill="#8b5a2b", anchor="mm")

    canvas.save(output_path)


def _crop_center(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = image.resize((math.ceil(src_w * scale), math.ceil(src_h * scale)))
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def parse_date_input(value: Optional[str]) -> date:
    if not value:
        return datetime.now().date()
    return datetime.strptime(value, "%Y-%m-%d").date()
