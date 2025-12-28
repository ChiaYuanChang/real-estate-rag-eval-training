from services.property_search_recommendation.chains import get_extract_user_question_intent_chain
from services.property_search_recommendation.models import RealEstateQuery


async def extract_user_question_intent(user_query: str) -> RealEstateQuery:
    chain = get_extract_user_question_intent_chain()
    response: RealEstateQuery = await chain.ainvoke(
        {
            "user_query": user_query
        }
    )
    return response
