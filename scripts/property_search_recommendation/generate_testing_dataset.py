import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

from config import GEMINI_AI_STUDIO_API_KEY, GEMINI_LLM_MODEL


def generate_testing_dataset(house_info: str) -> str:
    client = genai.Client(
        api_key=GEMINI_AI_STUDIO_API_KEY,
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""\
Here is the property metadata dataset:
<house_info>
{house_info}
<house_info/>"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            properties={
                "question_list": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "question": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "reason": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "type": genai.types.Schema(
                                type=genai.types.Type.STRING,
                                enum=["simple", "abstract"],
                            ),
                        },
                    ),
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text="""\
# SYSTEM PROMPT

## Context
You are an expert Data Scientist and QA Engineer specializing in Real Estate RAG (Retrieval-Augmented Generation) systems for the Taiwanese market. You possess deep knowledge of:
1.  **Taiwanese Real Estate Terminology:** (e.g., "透天", "平車", "三角窗", "正兩房", "衛浴開窗").
2.  **Kaohsiung Geography:** Key districts (Nanzi, Lingya, Zuoying, etc.) and landmarks (TSMC factory, MRT stations).
3.  **Search Behavior:** How Taiwanese users phrase housing needs, ranging from specific keyword searches to vague lifestyle descriptions.

## Objective
Your task is to generate a high-quality "Golden Dataset" for testing the recall accuracy of a property search engine. You must generate **10 testing queries** (5 Simple, 5 Abstract) based *strictly* on the provided property metadata.

**CRITICAL REQUIREMENT: High Information Density**
To ensure the target property is retrieved in the Top-K results, you must avoid generic queries.
-   **Stack Attributes:** Do not just ask for "3 bedrooms". Ask for "3 bedrooms + high floor + specific view + specific renovation details".
-   **Unique Anchors:** Identify unique keywords in the `tags`, `raw_description`, or `extracted_feature_list` (e.g., "fire sprinkler replaced", "pink wardrobe", "charging parking spot") and include them in the query.

## Style
Construct the queries using the following logic:

### 1. Simple Queries (Direct & Specific)
-   **Structure:** Explicitly combine 3-5 specific constraints.
-   **Formula:** [Location/Landmark] + [Price Range] + [Property Type] + [Specific Furniture/Renovation Detail].
-   **Example:** "I want a condo in Nanzi near the MRT, budget around 10 million, must have a kitchen island and a designated space for a dishwasher."

### 2. Abstract Queries (Persona & Scenario)
-   **Structure:** Create a specific user persona with a problem or desire that maps to specific property tags.
-   **Formula:** [Persona/Job] + [Pain Point/Desire] + [Implied Feature Requirement].
-   **Example:** "I work at TSMC and work late shifts. I need a place that is quiet, has blackout curtains or soundproof windows, and is move-in ready with all furniture included so I don't have to deal with renovations."

## Tone
-   **Queries:** Authentic Traditional Chinese (Taiwanese Mandarin). Use natural phrasing, local slang, and realistic sentence structures.
-   **Reasoning/Analysis:** Technical, objective, and analytical (in the `reason` field).

## Audience
The output is for the RAG Engineering Team. They use this data to verify if the embedding model successfully captures the semantic relationship between the long-tail keywords in the query and the metadata in the documents."""),
        ],
    )

    response = client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=contents,
        config=generate_content_config,
    )
    return response.text


def process_single_file(file_path: str):
    try:
        filename = os.path.basename(file_path)

        # 1. 讀取原始資料
        with open(file_path, "r", encoding="utf-8") as f:
            source_data = json.load(f)
            # 兼容有些 json 結構可能是直接物件，或是包在 listing 裡
            data = source_data.get("listing", source_data)

        house_info_parts: list[str] = []
        for label, key in field_mapping:
            val = data.get(key, "N/A")
            house_info_parts.append(f"{label}: {val}")
        property_id: str = data["property_id"]
        house_info = "\n".join(house_info_parts)

        # 3. 呼叫生成函數 (這裡假設 generate_testing_dataset 已經被 import 或定義)
        # 這裡會回傳 str 格式的 JSON
        response_str = generate_testing_dataset(house_info=house_info)

        # 4. 解析並儲存結果
        # 嘗試將 string 轉回 json object 以確保格式正確 (並能排版美觀)
        try:
            response_json = json.loads(response_str)
            response_json["property_id"] = property_id
        except json.JSONDecodeError:
            print(f"[Warning] Response for {filename} is not valid JSON. Saving as raw text.")
            response_json = {"property_id": property_id, "raw_content": response_str}

        output_file_path = os.path.join(save_response_folder_path, filename)

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(response_json, f, ensure_ascii=False, indent=2)

        return f"[Success] {filename}"

    except Exception as e:
        return f"[Error] {file_path}: {str(e)}"


if __name__ == "__main__":
    target_folder_path = "../../data/twhg_with_latlng_and_places/"
    save_response_folder_path = "../../data/testing_dataset_twhg_with_latlng_and_places/"

    field_mapping = [
        ("property_id", "property_id"),
        ("title", "title"),
        ("description", "description"),
        ("total_price", "total_price"),
        ("degree_of_decoration", "degree_of_decoration"),
        ("city", "city"),
        ("district", "district"),
        ("street", "street"),
        ("floor", "floor"),
        ("total_floors", "total_floors"),
        ("property_type", "property_type"),
        ("property_usage", "property_usage"),
        ("property_age", "property_age"),
        ("gross_area", "gross_area"),
        ("interior_area", "interior_area"),
        ("exclusive_use_area", "exclusive_use_area"),
        ("common_area", "parking_area"),
        ("num_bedroom", "num_bedroom"),
        ("num_bathroom", "num_bathroom"),
        ("num_living_room", "num_living_room"),
        ("orientation", "orientation"),
    ]

    # get all json file
    json_files = [
        os.path.join(target_folder_path, f)
        for f in os.listdir(target_folder_path)
        if f.endswith('.json')
    ]

    total_files = len(json_files)
    BATCH_SIZE = 10
    print(f"Found {total_files} files. Starting batch processing with size {BATCH_SIZE}...")
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = {executor.submit(process_single_file, file_path): file_path for file_path in json_files}
        completed_count = 0
        for future in as_completed(futures):
            result = future.result()
            completed_count += 1
            print(f"[{completed_count}/{total_files}] {result}")
