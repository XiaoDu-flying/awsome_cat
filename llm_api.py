import base64
import json
import os
from typing import Any, Optional

from openai import AzureOpenAI, OpenAI


DEFAULT_API_VERSION = "2024-02-01"
DEFAULT_ENDPOINT = "https://aidp.bytedance.net/api/modelhub/online/v2/crawl"
DEFAULT_MODEL = "gpt-5.5-2026-04-24"


def get_client() -> Optional[object]:
    """按环境变量创建 OpenAI/AzureOpenAI 客户端。"""

    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_AZURE_ENDPOINT")
    openai_base_url = os.getenv("OPENAI_BASE_URL")

    if azure_api_key and azure_endpoint:
        return AzureOpenAI(
            api_key=azure_api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
            azure_endpoint=azure_endpoint,
            default_headers={"X-TT-LOGID": os.getenv("X_TT_LOGID", "awesome-cat-local")},
        )

    if openai_api_key:
        client_kwargs: dict[str, Any] = {"api_key": openai_api_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        return OpenAI(**client_kwargs)

    return None


def get_default_model() -> str:
    return os.getenv("LLM_MODEL") or os.getenv("AZURE_OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL


def _extract_text_content(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text", ""))
        return "\n".join(part for part in parts if part)
    return str(content)


def chat_json(
    user_prompt: str,
    *,
    system_prompt: str,
    image_bytes: Optional[bytes] = None,
    image_mime_type: str = "image/png",
    model: Optional[str] = None,
    max_tokens: int = 4000,
) -> Optional[dict[str, Any]]:
    """调用多模态模型并尽量解析为 JSON，失败时返回 None。"""

    client = get_client()
    if client is None:
        return None

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]

    if image_bytes:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime_type};base64,{encoded}"},
            }
        )

    try:
        response = client.chat.completions.create(
            model=model or get_default_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=max_tokens,
            stream=False,
        )
        raw_text = _extract_text_content(response.choices[0].message).strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            raw_text = raw_text.replace("json\n", "", 1).strip()
        return json.loads(raw_text)
    except Exception:
        return None


def chat_text(
    user_prompt: str,
    *,
    system_prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 1200,
) -> Optional[str]:
    """调用文本模型，失败时返回 None。"""

    client = get_client()
    if client is None:
        return None

    try:
        response = client.chat.completions.create(
            model=model or get_default_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            stream=False,
        )
        return _extract_text_content(response.choices[0].message).strip()
    except Exception:
        return None


if __name__ == "__main__":
    answer = chat_text(
        "请用一句话介绍这个模块的作用。",
        system_prompt="你是一个简洁的 Python 开发助手。",
        max_tokens=200,
    )
    print(answer or "LLM API 不可用，请检查网络或配置。")
