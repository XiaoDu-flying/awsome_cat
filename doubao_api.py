import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI


# DEFAULT_IMAGE_PATH = "/Users/bytedance/Library/Caches/coco/sessions/eac6f23e-019c-4b95-bf29-0a6f3f85747d/file-cache/file_1778662710.jpg"
DEFAULT_IMAGE_PATH = "/Users/bytedance/Documents/lxh作业/awesome_cat/uploads/a8057e67301377e7ad201c338ed31412.jpg"
PROMPT = "你看见了什么？"


def build_image_url(image_source: str) -> str:
    """将远程 URL 或本地图片路径转换为接口可用的 image_url。"""
    if image_source.startswith(("http://", "https://", "data:")):
        return image_source

    image_path = Path(image_source).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


def main() -> None:
    # 从环境变量中获取您的 API KEY，配置方法见：https://www.volcengine.com/docs/82379/1399008
    api_key = os.getenv("ARK_API_KEY") or "ark-a2e949f5-1f40-4a02-8f36-abb58f1d9e65-fcb08"
    image_source = os.getenv("DOUBAO_IMAGE_PATH", DEFAULT_IMAGE_PATH)

    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )

    response = client.responses.create(
        model="doubao-seed-1-6-flash-250828",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": build_image_url(image_source),
                    },
                    {
                        "type": "input_text",
                        "text": PROMPT,
                    },
                ],
            }
        ],
    )

    print(response.output_text)


if __name__ == "__main__":
    main()
