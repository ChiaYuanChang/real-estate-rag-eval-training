import asyncio
import numpy as np
from openai import AsyncOpenAI

from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from services.property_search_recommendation.models import CypherVariables, RealEstateQuery
from infrastructures.neo4j.client import neo4j_client

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

FILTER_CYPHER = """
MATCH (p:Property)
WHERE
  ($city IS NULL OR p.city = $city) AND
  ($district IS NULL OR p.district = $district) AND
  ($street IS NULL OR p.street CONTAINS $street) AND
  ($min_price IS NULL OR p.total_price >= $min_price) AND
  ($max_price IS NULL OR p.total_price <= $max_price) AND
  ($min_interior_area IS NULL OR p.interior_area >= $min_interior_area) AND
  ($min_bedroom IS NULL OR p.num_bedroom >= $min_bedroom) AND
  ($min_bathroom IS NULL OR p.num_bathroom >= $min_bathroom) AND
  ($property_type IS NULL OR p.property_type = $property_type) AND
  ($min_age IS NULL OR p.property_age >= $min_age) AND
  ($max_age IS NULL OR p.property_age <= $max_age)

RETURN
  p.property_id AS property_id,
  p.title AS title,
  p.total_price AS total_price,
  p.text_embedding AS embedding
LIMIT $limit;
"""


async def embed(text: str):
    resp = await client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=[text])
    return np.array(resp.data[0].embedding, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # assume embeddings are not normalized; normalize safely
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def build_query_text(abstract_requirements: list[str]) -> str:
    # You can enrich this string
    return "需求：" + "；".join([x.strip() for x in abstract_requirements if x.strip()])


async def hybrid_search(query: RealEstateQuery, graph_limit=200, topk=10):
    # 1) Graph filter
    rows = await neo4j_client.query(FILTER_CYPHER, **query.cypher_variables.model_dump(), limit=graph_limit)

    if not rows:
        return []

    # 2) Embedding rerank
    q_text = build_query_text(query.abstract_requirements)
    q_emb = await embed(q_text)

    scored = []
    for r in rows:
        emb = r.get("embedding")
        if not emb:
            continue
        p_emb = np.array(emb, dtype=np.float32)
        score = cosine_sim(q_emb, p_emb)
        scored.append({
            "property_id": r["property_id"],
            "title": r["title"],
            "total_price": r["total_price"],
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:topk]
