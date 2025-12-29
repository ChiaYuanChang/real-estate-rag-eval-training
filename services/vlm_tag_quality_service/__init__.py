from services.vlm_tag_quality_service.chains import get_vlm_as_a_judge_chain
from services.vlm_tag_quality_service.models import RealEstateTagEvaluation


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
