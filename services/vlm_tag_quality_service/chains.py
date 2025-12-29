from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, OPENAI_LLM_MODEL
from services.vlm_tag_quality_service.models import RealEstateTagEvaluation, SpatialEvaluation
from services.vlm_tag_quality_service.prompts import (
    VLM_AS_A_JUDGE_SYSTEM_PROMPT,
    VLM_AS_A_JUDGE_USER_PROMPT,
    VLM_WITH_SPATIAL_SIGNALS_SYSTEM_PROMPT,
    VLM_WITH_SPATIAL_SIGNALS_USER_PROMPT,
)

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_LLM_MODEL,
    temperature=0
)


def get_vlm_as_a_judge_chain():
    structured_llm = llm.with_structured_output(RealEstateTagEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", VLM_AS_A_JUDGE_SYSTEM_PROMPT),
        ("human", [
            {"type": "text", "text": VLM_AS_A_JUDGE_USER_PROMPT},
            {"type": "image_url", "image_url": {"url": "{image_url}"}},
        ])
    ])

    chain = prompt | structured_llm
    return chain


def get_vlm_with_spatial_signals_chain():
    structured_llm = llm.with_structured_output(SpatialEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", VLM_WITH_SPATIAL_SIGNALS_SYSTEM_PROMPT),
        ("human", [
            {"type": "text", "text": VLM_WITH_SPATIAL_SIGNALS_USER_PROMPT},
            {"type": "image_url", "image_url": {"url": "{image_url}"}},
            {"type": "image_url", "image_url": {"url": "{spatial_signal_image_url}"}},
        ])
    ])

    chain = prompt | structured_llm
    return chain
