from typing import Any, List

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from tqdm import tqdm

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

# batch size for OpenAI embeddings
BATCH_SIZE = 64
load_dotenv()
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Cypher
# -----------------------------
GET_PROPERTIES_NO_EMBED = """
MATCH (p:Property)
WHERE p.text_embedding IS NULL
RETURN p.property_id AS property_id,
       coalesce(p.title,'') + '\n' +
       coalesce(p.description,'') + '\n' +
       coalesce(p.raw_description,'') AS text
"""

SET_EMBEDDING = """
MATCH (p:Property {property_id: $property_id})
SET p.text_embedding = $embedding
"""


def chunk_list(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Call OpenAI Embeddings API for a list of texts (batch).
    Docs: https://platform.openai.com/docs/guides/embeddings
    """
    # The embeddings API accepts a list of strings as input.  [oai_citation:3‡OpenAI Platform](https://platform.openai.com/docs/api-reference/embeddings/create?lang=python&utm_source=chatgpt.com)
    resp = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts
    )
    # resp.data is list aligned with inputs
    return [d.embedding for d in resp.data]


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            rows = session.run(GET_PROPERTIES_NO_EMBED).data()

        if not rows:
            print("✅ No properties need embeddings (all have text_embedding).")
            return

        print(f"Found {len(rows)} properties without embedding.")
        print(f"Embedding model: text-embedding-3-small")

        # Prepare data
        items = []
        for r in rows:
            pid = r["property_id"]
            text = (r["text"] or "").strip()
            # Avoid empty strings (OpenAI embeddings input cannot be empty)  [oai_citation:4‡OpenAI Platform](https://platform.openai.com/docs/api-reference/embeddings/create?lang=python&utm_source=chatgpt.com)
            if not text:
                text = "(empty)"
            items.append({"property_id": pid, "text": text})

        updated = 0

        for batch in tqdm(list(chunk_list(items, BATCH_SIZE))):
            texts = [x["text"] for x in batch]
            pids = [x["property_id"] for x in batch]

            embeddings = get_embeddings(texts)

            # write back to neo4j
            with driver.session() as session:
                for pid, emb in zip(pids, embeddings):
                    session.run(SET_EMBEDDING, property_id=pid, embedding=emb).consume()
                    updated += 1

        print("===================================")
        print(f"✅ Embedded & updated: {updated}")
        print("===================================")
        print("Done.")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
