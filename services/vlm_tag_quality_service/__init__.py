import base64
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import pipeline

from services.vlm_tag_quality_service.chains import get_vlm_as_a_judge_chain
from services.vlm_tag_quality_service.models import RealEstateTagEvaluation

model_id = "depth-anything/Depth-Anything-V2-Large-hf"

pipe = pipeline(
    task="depth-estimation",
    model=model_id,
    device=0,  # GPU
    torch_dtype=torch.float16
)


async def using_vlm_as_a_judge(room_name: str, tag_info: str, raw_description: str, image_url: str) -> RealEstateTagEvaluation:
    chain = get_vlm_as_a_judge_chain()
    response: RealEstateTagEvaluation = await chain.ainvoke(
        {
            "room_name": room_name,
            "tag_info": tag_info,
            "raw_description": raw_description,
            "image_url": image_url,
        }
    )
    return response


def _depth_anything_v2_pipeline(image_url: str):
    # 下載圖片
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")

    # 推論
    out = pipe(image)

    return out["depth"]  # PIL.Image


def _pil_to_base64_url(img: Image.Image, format="PNG") -> str:
    buffered = BytesIO()

    img.save(buffered, format=format)

    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:image/{format.lower()};base64,{img_str}"


async def using_vlm_with_spatial_signals_info_as_a_judge(room_name: str, tag_info: str, raw_description: str,
                                                         image_url: str, ) -> RealEstateTagEvaluation:
    chain = get_vlm_as_a_judge_chain()
    depth_map = _depth_anything_v2_pipeline(image_url)

    response: RealEstateTagEvaluation = await chain.ainvoke(
        {
            "room_name": room_name,
            "tag_info": tag_info,
            "raw_description": raw_description,
            "image_url": image_url,
            "spatial_signal_image_url": _pil_to_base64_url(depth_map)
        }
    )
    return response
