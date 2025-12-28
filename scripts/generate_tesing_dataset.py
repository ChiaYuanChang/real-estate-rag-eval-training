import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

from config import GEMINI_AI_STUDIO_API_KEY


def generate_testing_dataset(house_info: str) -> str:
    client = genai.Client(
        api_key=GEMINI_AI_STUDIO_API_KEY,
    )

    model = "gemini-flash-latest"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""\
Here is a set of house information in <house_info>. Please help me convert the following information into the required format.
Generate 10 testing queries (5 Simple, 5 Abstract) based *only* on the logic found in these files.

**Instructions for Query Design:**
1. **Simple Queries:** focus on combining 2-3 specific attributes (e.g., "Nanzi district + 3 bedrooms + bright light").
2. **Abstract Queries:** create a persona or a problem.
   - *Example:* If a property has a "no window in bathroom" tag, the query could be "I hate mold, do you have good ventilation?" (to test if the system avoids or warns about that property, or finds one with a window).
   - *Example:* If a property has a "bowling alley", the query could be "I want a place where I can play sports indoors with friends."
   - *Example:* If a property has "open kitchen", the query could be "I want to chat with my family while cooking."

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
## Context
You are an expert Data Scientist and QA Engineer specializing in Real Estate RAG (Retrieval-Augmented Generation) systems. You are tasked with generating high-quality testing datasets to evaluate the retrieval accuracy and semantic understanding of a property search engine. You understand Taiwanese real estate terminology, geography (specifically Kaohsiung), and cultural nuances in how users describe housing needs.

## Objective
Generate a set of 10 distinct testing queries based strictly on the provided JSON property data.
The output must be split into two categories:
1. **5 Simple Queries:** Direct, keyword-heavy requests focusing on specific features (e.g., location, price, layout, specific tags).
2. **5 Abstract Queries:** Vague, emotional, or scenario-based requests where the user implies a need rather than stating it directly (e.g., health concerns implying ventilation needs, lifestyle descriptions implying specific amenities).

## Style
The output must be structured as a JSON list or a clear Markdown list.
Each entry must contain:
- `query`: The user's question in Traditional Chinese (Taiwanese Mandarin).
- `target_property_id`: The ID of the property that best matches this query.
- `type`: \"Simple\" or \"Abstract\".
- `reason`: A brief explanation (<150 words) of why this query was designed and what specific aspect of the RAG system it tests (e.g., keyword matching, semantic inference, negations).

## Tone
- **Queries:** Varied tone mimicking real users. Some should be concise/robotic, others conversational/confused, and some demanding/emotional.
- **Reasoning:** Analytical, technical, and objective.

## Audience
The RAG system development team who needs to verify if the embedding model and retrieval logic are working correctly."""),
        ],
    )

    response = client.models.generate_content(
        model=model,
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
    target_folder_path = "../data/twhg_with_latlng_and_places/"
    save_response_folder_path = "../data/testing_dataset_twhg_with_latlng_and_places/"

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
