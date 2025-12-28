import os
import json
import glob
from typing import Any, Dict, List, Optional, Tuple
from neo4j import GraphDatabase
from tqdm import tqdm


NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

DATA_DIR = "../../data/cleaned_twhg_with_latlng_and_places/"


CONSTRAINTS_AND_INDEXES = [
    """
    CREATE CONSTRAINT property_id_unique IF NOT EXISTS
    FOR (p:Property) REQUIRE p.property_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT city_name_unique IF NOT EXISTS
    FOR (c:City) REQUIRE c.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT district_key_unique IF NOT EXISTS
    FOR (d:District) REQUIRE d.key IS UNIQUE
    """,
    """
    CREATE CONSTRAINT street_key_unique IF NOT EXISTS
    FOR (s:Street) REQUIRE s.key IS UNIQUE
    """,
    """
    CREATE CONSTRAINT tag_name_unique IF NOT EXISTS
    FOR (t:Tag) REQUIRE t.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT room_name_unique IF NOT EXISTS
    FOR (r:Room) REQUIRE r.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT property_type_unique IF NOT EXISTS
    FOR (pt:PropertyType) REQUIRE pt.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT image_url_unique IF NOT EXISTS
    FOR (img:Image) REQUIRE img.url IS UNIQUE
    """,
    # Useful indexes for filtering
    "CREATE INDEX property_price IF NOT EXISTS FOR (p:Property) ON (p.total_price)",
    "CREATE INDEX property_city IF NOT EXISTS FOR (p:Property) ON (p.city)",
    "CREATE INDEX property_district IF NOT EXISTS FOR (p:Property) ON (p.district)",
    "CREATE INDEX property_type IF NOT EXISTS FOR (p:Property) ON (p.property_type)",
]

# Cypher: check exists
CHECK_EXISTS = """
MATCH (p:Property {property_id: $property_id})
RETURN p.property_id AS property_id
LIMIT 1
"""

# Cypher: import one property (MERGE)
IMPORT_ONE = """
// --- Property ---
MERGE (p:Property {property_id: $property_id})
SET
  p.title = $title,
  p.total_price = $total_price,
  p.property_type = $property_type,
  p.property_age = $property_age,
  p.gross_area = $gross_area,
  p.interior_area = $interior_area,
  p.public_area_ratio = $public_area_ratio,
  p.num_bedroom = $num_bedroom,
  p.num_bathroom = $num_bathroom,
  p.num_living_room = $num_living_room,
  p.floor = $floor,
  p.total_floors = $total_floors,
  p.land_ownership_area = $land_ownership_area,
  p.property_usage = $property_usage,
  p.orientation = $orientation,
  p.original_url = $original_url,
  p.description = $description,
  p.raw_description = $raw_description,
  p.city = $city,
  p.district = $district,
  p.street = $street

// Location hierarchy
MERGE (c:City {name: $city})
MERGE (d:District {key: $district_key})
  ON CREATE SET d.name = $district
MERGE (s:Street {key: $street_key})
  ON CREATE SET s.name = $street

MERGE (d)-[:IN_CITY]->(c)
MERGE (s)-[:IN_DISTRICT]->(d)
MERGE (p)-[:LOCATED_ON]->(s)

// Property Type
MERGE (pt:PropertyType {name: $property_type})
MERGE (p)-[:HAS_TYPE]->(pt)

// Rooms & Tags
WITH p
UNWIND $extracted_feature_list AS rf
  MERGE (r:Room {name: rf.room})
  MERGE (p)-[:HAS_ROOM]->(r)
  WITH p, r, rf
  UNWIND rf.tag_list AS tagName
    MERGE (t:Tag {name: tagName})
    MERGE (r)-[:HAS_TAG]->(t)
    MERGE (p)-[:HAS_TAG]->(t)

// Images
WITH p
UNWIND $picture_list AS imgUrl
  MERGE (img:Image {url: imgUrl})
  MERGE (p)-[:HAS_IMAGE]->(img)

RETURN p.property_id AS property_id
"""


# Helpers
def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def normalize_extracted_feature_list(raw: Any) -> List[Dict[str, Any]]:
    """
    Ensure extracted_feature_list is always:
    [{"room": "...", "tag_list": ["...", ...]}, ...]
    """
    if not raw:
        return []
    if isinstance(raw, list):
        out = []
        for item in raw:
            room = item.get("room") if isinstance(item, dict) else None
            tags = item.get("tag_list") if isinstance(item, dict) else None
            if not room:
                continue
            if not tags:
                tags = []
            # ensure list[str]
            tags = [str(t).strip() for t in tags if str(t).strip()]
            out.append({"room": str(room).strip(), "tag_list": tags})
        return out
    return []


def normalize_picture_list(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(u).strip() for u in raw if str(u).strip()]
    return []


def make_location_keys(city: str, district: str, street: str) -> Tuple[str, str]:
    """
    Use composite keys to avoid name collision (e.g. '中山路').
    """
    district_key = f"{city}|{district}"
    street_key = f"{city}|{district}|{street}"
    return district_key, street_key


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_params(doc: Dict[str, Any]) -> Dict[str, Any]:
    property_id = str(doc.get("property_id", "")).strip()
    if not property_id:
        raise ValueError("Missing property_id")

    city = str(doc.get("city", "")).strip()
    district = str(doc.get("district", "")).strip()
    street = str(doc.get("street", "")).strip()
    district_key, street_key = make_location_keys(city, district, street)

    params = {
        "property_id": property_id,
        "title": doc.get("title"),
        "total_price": safe_float(doc.get("total_price")),
        "property_type": doc.get("property_type"),
        "property_age": safe_int(doc.get("property_age")),
        "gross_area": safe_float(doc.get("gross_area")),
        "interior_area": safe_float(doc.get("interior_area")),
        "public_area_ratio": safe_float(doc.get("public_area_ratio")),
        "num_bedroom": safe_int(doc.get("num_bedroom")),
        "num_bathroom": safe_int(doc.get("num_bathroom")),
        "num_living_room": safe_int(doc.get("num_living_room")),
        "floor": safe_int(doc.get("floor")),
        "total_floors": safe_int(doc.get("total_floors")),
        "land_ownership_area": safe_float(doc.get("land_ownership_area")),
        "property_usage": doc.get("property_usage"),
        "orientation": doc.get("orientation"),
        "original_url": doc.get("original_url"),
        "description": doc.get("description"),
        "raw_description": doc.get("raw_description"),
        "city": city,
        "district": district,
        "street": street,
        "district_key": district_key,
        "street_key": street_key,
        "extracted_feature_list": normalize_extracted_feature_list(doc.get("extracted_feature_list")),
        "picture_list": normalize_picture_list(doc.get("picture_list")),
    }
    return params


# Main import
def ensure_schema(driver):
    with driver.session() as session:
        for q in CONSTRAINTS_AND_INDEXES:
            session.run(q)


def property_exists(session, property_id: str) -> bool:
    rec = session.run(CHECK_EXISTS, property_id=property_id).single()
    return rec is not None


def import_one(session, params: Dict[str, Any]):
    session.run(IMPORT_ONE, **params).consume()


def main():
    json_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if not json_files:
        print(f"[ERROR] No .json files found in: {DATA_DIR}")
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        print("[1/3] Ensuring schema (constraints/indexes)...")
        ensure_schema(driver)

        total = 0
        imported = 0
        skipped = 0
        failed = 0

        print("[2/3] Importing JSON files...")
        with driver.session() as session:
            for path in tqdm(json_files):
                total += 1
                try:
                    doc = load_json(path)
                    params = build_params(doc)
                    pid = params["property_id"]

                    if property_exists(session, pid):
                        skipped += 1
                        continue

                    import_one(session, params)
                    imported += 1

                except Exception as e:
                    failed += 1
                    tqdm.write(f"[FAILED] {os.path.basename(path)} -> {e}")

        print("[3/3] Done.")
        print("===================================")
        print(f"Total scanned : {total}")
        print(f"Imported      : {imported}")
        print(f"Skipped       : {skipped}")
        print(f"Failed        : {failed}")
        print("===================================")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
