import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from google import genai
from google.genai import types

from config import GEMINI_AI_STUDIO_API_KEY

INPUT_DIR = "../../data/cleaned_twhg_with_latlng_and_places/"
OUTPUT_DIR = "../../data/vlm_rematch_twhg_with_latlng_and_places/"


def image_description_rematching(image_bytes_data_list: list[bytes], object_description: str):
    client = genai.Client(
        api_key=GEMINI_AI_STUDIO_API_KEY,
    )

    model = "gemini-flash-latest"
    contents = [
        types.Content(
            role="user",
            parts=[
                      types.Part.from_text(text=f"""\
## Input Data

### 1. Object Descriptions (Targets)
Below is the list of descriptions to be matched:
```object description
{object_description}
```

### 2. Images (Candidates)
I am providing {len(image_bytes_data_list)} images. Please refer to the first image uploaded/attached as [Image_1], the second as [Image_2], and so on, following the natural sequence of the input.

---

## Instructions
1. **Analyze All Candidates**: Scan all images to understand the global context and identify unique features.
2. **Eliminate Redundancy**: Determine which images do not fit any of the descriptions.
3. **Global Optimization**: Ensure that assigning a description to an image doesn't prevent a "better" match for a subsequent description (Logical Mutual Exclusion).
4. **Format Output**: Provide the result as follows:

**Reasoning Log:**
(Provide a brief explanation of the matching logic and how you resolved any visual ambiguities.)

**Matching Results:**
```json
[
  {{
    "description_id": "Desc_01",
    "matched_image": "Image_X",
    "evidence": "Briefly state why this image was chosen over others.",
    "confidence_score": 0.98
  }},
  ...
]"""),
                  ] + image_bytes_data_list,
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
                "reasoning_log": genai.types.Schema(
                    type=genai.types.Type.STRING,
                ),
                "matching_result_list": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "description_id": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "matched_image": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "evidence": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "confidence_score": genai.types.Schema(
                                type=genai.types.Type.NUMBER,
                            ),
                        },
                    ),
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text="""\
## Context
You are a highly skilled Multimodal Data Scientist specializing in computer vision and semantic alignment. You are tasked with rectifying a corrupted dataset where images and their corresponding textual descriptions have been shuffled. The dataset contains up to 20 images and a smaller or equal number of object descriptions. Your goal is to restore the 1:1 mapping between descriptions and the correct images.

## Objective
Identify the single best-matching image for each provided description.
1. Perform a 1:1 match for every description (every description MUST be mapped to exactly one image).
2. Since there are more images than descriptions, identify which images are \"redundant\" (have no matching description).
3. Use visual cues such as color, texture, spatial relationships, OCR text, and object categories to differentiate between similar candidates.

## Style
Structured, analytical, and evidence-based. You must use a \"Chain-of-Thought\" approach: analyze the visual evidence for all candidate images before finalizing the match.

## Tone
Professional, objective, and precise. If a match is ambiguous, explain the reasoning but still provide the most statistically likely assignment.

## Audience
Technical data engineers who will use your output to programmatically re-label the dataset.

## Response
Provide your response in two distinct parts:
1. **Reasoning Log**: A brief analytical breakdown of how you distinguished between similar images.
2. **Final Mapping**: A strictly formatted JSON array containing the results."""),
        ],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return response.text


def format_object_description(extracted_feature_list: list[dict]) -> str:
    extracted_feature_id = 1
    total_description: str = ""
    for extracted_feature in extracted_feature_list:
        room_name = extracted_feature.get('room', '')

        tags = extracted_feature.get('tag_list', [])

        formatted_tags = ",".join([f'"{tag}"' for tag in tags])

        total_description += (
            f"[Desc_{extracted_feature_id}]"
            f"<room_name>{room_name}<room_name/>"
            f"<room_tags>{formatted_tags}<room_tags/>"
        )
        extracted_feature_id += 1
    return total_description


def image_url_list_to_bytes(image_url_list: list[str]) -> tuple[list, list[str]]:
    image_bytes_data_list: list = []
    original_image_url_list: list[str] = []
    for image_url in image_url_list:
        response = requests.get(image_url)
        if response.status_code == 200:
            mime_type = response.headers.get('Content-Type', 'image/jpeg')
            image_bytes_data_list.append(
                types.Part.from_bytes(
                    data=response.content,
                    mime_type=mime_type
                )
            )
            original_image_url_list.append(image_url)
    return image_bytes_data_list, original_image_url_list


def replace_image_placeholders(json_data: dict, url_list: list, extracted_feature_list: list[dict]):
    processed_list = []

    for idx, item in enumerate(json_data['matching_result_list']):
        new_item = item.copy()
        image_key = new_item['matched_image']

        if image_key.startswith("Image_"):
            try:
                index_str = image_key.split('_')[1]
                index = int(index_str) - 1

                if 0 <= index < len(url_list):
                    new_item['matched_image'] = url_list[index]
                else:
                    print(f"Warning: {image_key} corresponds to index {index}, which is out of range of the URL list.")
            except ValueError:
                print(f"Warning: Unable to parse image identifier {image_key}")

        new_item['extracted_feature'] = extracted_feature_list[idx]
        processed_list.append(new_item)

    json_data['matching_result_list'] = processed_list
    return json_data


def process_single_file(file_path: str):
    try:
        filename = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        image_bytes_data_list, original_image_url_list = image_url_list_to_bytes(image_url_list=json_data["picture_list"])
        description = format_object_description(extracted_feature_list=json_data["extracted_feature_list"])
        response_str: str = image_description_rematching(
            image_bytes_data_list=image_bytes_data_list,
            object_description=description
        )
        response_json = replace_image_placeholders(json_data=json.loads(response_str), url_list=original_image_url_list,
                                                   extracted_feature_list=json_data["extracted_feature_list"])

        output_file_path = os.path.join(OUTPUT_DIR, filename)

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(response_json, f, ensure_ascii=False, indent=2)

        return f"[Success] {filename}"

    except Exception as e:
        return f"[Error] {file_path}: {str(e)}"


if __name__ == "__main__":
    json_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
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
