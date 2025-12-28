from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, OPENAI_LLM_MODEL
from services.property_search_recommendation.models import RealEstateQuery
from services.property_search_recommendation.prompts import (
    EXTRACT_USER_QUESTION_INTENT_SYSTEM_PROMPT,
    EXTRACT_USER_QUESTION_INTENT_USER_PROMPT,
)

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_LLM_MODEL,
    temperature=0
)


def get_extract_user_question_intent_chain():
    structured_llm = llm.with_structured_output(RealEstateQuery)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACT_USER_QUESTION_INTENT_SYSTEM_PROMPT),
        ("human", EXTRACT_USER_QUESTION_INTENT_USER_PROMPT),
    ])

    chain = prompt | structured_llm
    return chain
