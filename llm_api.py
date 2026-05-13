import base64
import json
import os
import threading
from typing import Any, Optional

from openai import AzureOpenAI, OpenAI


DEFAULT_API_VERSION = "2024-02-01"
DEFAULT_ENDPOINT = "https://aidp.bytedance.net/api/modelhub/online/v2/crawl"
DEFAULT_MODEL = "gpt-5.5-2026-04-24"
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_MODEL = "doubao-seed-1-8-251228"
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_RETRIES = 2
_THREAD_STATE = threading.local()


def _set_last_llm_diagnostic(message: str) -> None:
    _THREAD_STATE.last_llm_diagnostic = message


def get_last_llm_diagnostic() -> str:
    return getattr(_THREAD_STATE, "last_llm_diagnostic", "")


def _get_runtime_options() -> tuple[float, int, str, str]:
    timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
    ark_base_url = os.getenv("ARK_BASE_URL", DEFAULT_ARK_BASE_URL)
    model = get_default_model()
    return timeout_seconds, max_retries, ark_base_url, model


def _runtime_summary() -> str:
    timeout_seconds, max_retries, ark_base_url, model = _get_runtime_options()
    return (
        f"model={model}, timeout={timeout_seconds}s, max_retries={max_retries}, "
        f"ark_base_url={ark_base_url}"
    )


def get_llm_status() -> tuple[bool, str]:
    """返回当前是否已配置可用 LLM，以及对应的原因说明。"""

    if os.getenv("ARK_API_KEY"):
        return True, "已检测到 ARK_API_KEY，使用 ARK Doubao 模型。"
    if os.getenv("AZURE_OPENAI_API_KEY") and (os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_AZURE_ENDPOINT")):
        return True, "已检测到 Azure OpenAI 配置。"
    if os.getenv("OPENAI_API_KEY"):
        return True, "已检测到 OpenAI API Key。"
    return False, "未检测到 ARK_API_KEY / OPENAI_API_KEY / AZURE_OPENAI_API_KEY，已回退为模板生成。"


def get_client() -> Optional[object]:
    """按环境变量创建 OpenAI/AzureOpenAI 客户端。"""

    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    ark_api_key = os.getenv("ARK_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_AZURE_ENDPOINT")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    timeout_seconds, max_retries, ark_base_url, _ = _get_runtime_options()

    if ark_api_key:
        return OpenAI(
            api_key=ark_api_key,
            base_url=ark_base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    if azure_api_key and azure_endpoint:
        return AzureOpenAI(
            api_key=azure_api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
            azure_endpoint=azure_endpoint,
            default_headers={"X-TT-LOGID": os.getenv("X_TT_LOGID", "awesome-cat-local")},
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    if openai_api_key:
        client_kwargs: dict[str, Any] = {
            "api_key": openai_api_key,
            "timeout": timeout_seconds,
            "max_retries": max_retries,
        }
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        return OpenAI(**client_kwargs)

    return None


def get_default_model() -> str:
    if os.getenv("ARK_API_KEY"):
        return os.getenv("ARK_MODEL") or DEFAULT_ARK_MODEL
    return os.getenv("LLM_MODEL") or os.getenv("AZURE_OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL


def _build_data_url(image_bytes: bytes, image_mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{image_mime_type};base64,{encoded}"


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None)
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for block in content:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(text)
        if parts:
            return "\n".join(parts).strip()
    return ""


def _is_ark_client(client: object) -> bool:
    base_url = getattr(client, "base_url", None)
    return base_url is not None and DEFAULT_ARK_BASE_URL in str(base_url)


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
        _set_last_llm_diagnostic(f"未创建 LLM 客户端，请检查 API Key 与端点配置。({_runtime_summary()})")
        return None

    try:
        if _is_ark_client(client):
            # 与 doubao_api.py 保持一致：input 仅包含 user 消息，避免系统消息结构差异导致挂起
            user_content: list[dict[str, Any]] = [{"type": "input_text", "text": f"{system_prompt}\n\n{user_prompt}"}]
            if image_bytes:
                user_content.insert(
                    0,
                    {"type": "input_image", "image_url": _build_data_url(image_bytes, image_mime_type)},
                )

            response = client.responses.create(
                model=model or get_default_model(),
                input=[{"role": "user", "content": user_content}],
            )
            raw_text = _extract_response_text(response).strip()
        else:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            if image_bytes:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _build_data_url(image_bytes, image_mime_type)},
                    }
                )
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
        if not raw_text:
            _set_last_llm_diagnostic(f"LLM 已响应，但返回内容为空。({_runtime_summary()})")
            return None
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            raw_text = raw_text.replace("json\n", "", 1).strip()
        result = json.loads(raw_text)
        _set_last_llm_diagnostic("LLM 调用成功。")
        return result
    except Exception as exc:
        _set_last_llm_diagnostic(f"LLM 调用失败：{type(exc).__name__}: {exc} ({_runtime_summary()})")
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
        _set_last_llm_diagnostic(f"未创建 LLM 客户端，请检查 API Key 与端点配置。({_runtime_summary()})")
        return None

    try:
        if _is_ark_client(client):
            response = client.responses.create(
                model=model or get_default_model(),
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"{system_prompt}\n\n{user_prompt}"}
                        ],
                    }
                ],
            )
            text = _extract_response_text(response).strip()
            if not text:
                _set_last_llm_diagnostic(f"ARK 响应成功，但返回文本为空。({_runtime_summary()})")
                return None
            _set_last_llm_diagnostic("LLM 调用成功。")
            return text

        response = client.chat.completions.create(
            model=model or get_default_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            stream=False,
        )
        text = _extract_text_content(response.choices[0].message).strip()
        if not text:
            _set_last_llm_diagnostic(f"LLM 响应成功，但返回文本为空。({_runtime_summary()})")
            return None
        _set_last_llm_diagnostic("LLM 调用成功。")
        return text
    except Exception as exc:
        _set_last_llm_diagnostic(f"LLM 调用失败：{type(exc).__name__}: {exc} ({_runtime_summary()})")
        return None


if __name__ == "__main__":
    answer = chat_text(
        "请用一句话介绍这个模块的作用。",
        system_prompt="你是一个简洁的 Python 开发助手。",
        max_tokens=200,
    )
    print(answer or "LLM API 不可用，请检查网络或配置。")
