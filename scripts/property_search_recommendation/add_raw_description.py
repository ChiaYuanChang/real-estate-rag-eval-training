import glob
import json
import os
import re

ORIGINAL_DATA_DIR = "../../data/twhg_with_latlng_and_places/"
REMATCH_DATA_DIR = "../../data/vlm_rematch_twhg_with_latlng_and_places/"
OUTPUT_DIR = "../../data/vlm_rematch_add_info_twhg_with_latlng_and_places/"

# 使用與 clean_raw_data.py 相同的 Regex 邏輯來定位區塊
# Group 1: Room Name (e.g., 客廳/餐工作區)
# Group 2: Raw Tags Content (e.g., 格局方正(長方形空間好擺家具), 採光普通...)
PATTERN = re.compile(r"(?:^|\n)(?:Image\s*)?\d+\s+([^\n,]+).*?Tags:\s*([^\n]+)", re.IGNORECASE)


def extract_raw_description_map(description_text: str) -> dict:
    """
    解析原始描述文字，回傳一個字典:
    Key: Room Name (e.g., "客廳")
    Value: Raw Tags String (e.g., "採光尚可(兩扇窗戶引入日光), 格局狹長...")
    """
    mapping = {}
    if not description_text:
        return mapping

    for match in PATTERN.finditer(description_text):
        room_raw = match.group(1).strip()
        tags_raw = match.group(2).strip()

        # 將解析到的原始 Tag 字串存入 Map
        if room_raw and tags_raw:
            mapping[room_raw] = tags_raw

    return mapping


def process_single_file(rematch_file_path: str, filename: str):
    # 1. 讀取 Rematch 資料 (Target to update)
    with open(rematch_file_path, 'r', encoding='utf-8') as f:
        rematch_data = json.load(f)

    # 2. 讀取 Original 資料 (Source of raw text)
    original_file_path = os.path.join(ORIGINAL_DATA_DIR, filename)

    # 雖然你提到不會少檔案，但為了腳本健壯性，還是檢查一下
    if not os.path.exists(original_file_path):
        print(f"[Warning] Original file not found for {filename}, skipping.")
        return

    with open(original_file_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # 3. 從原始資料中提取 Description 並建立對照表
    raw_full_description = original_data.get('listing', {}).get('description', '')
    raw_desc_map = extract_raw_description_map(raw_full_description)

    # 4. 將 Raw Description 注入到 Rematch 資料中
    if 'matching_result_list' in rematch_data:
        for item in rematch_data['matching_result_list']:
            extracted_feature = item.get('extracted_feature', {})
            room_name = extracted_feature.get('room')

            if room_name and room_name in raw_desc_map:
                # 新增 raw_description 欄位
                extracted_feature['raw_description'] = raw_desc_map[room_name]
            else:
                # 如果找不到對應的原始描述 (極少數情況)，設為 None 或空字串
                extracted_feature['raw_description'] = None
                # print(f"[Info] No raw description found for room: {room_name} in {filename}")

    # 5. 寫入新檔案
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rematch_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 取得所有 Rematch 資料夾下的 json 檔案
    json_files = glob.glob(os.path.join(REMATCH_DATA_DIR, "*.json"))
    print(f"Found {len(json_files)} files in {REMATCH_DATA_DIR}")

    processed_count = 0

    for file_path in json_files:
        filename = os.path.basename(file_path)
        try:
            process_single_file(file_path, filename)
            processed_count += 1
        except Exception as e:
            print(f"[Error] Failed to process {filename}: {e}")

    print("-" * 30)
    print(f"Job Finished.")
    print(f"Total Processed: {processed_count}")
    print(f"Output Directory: {OUTPUT_DIR}")
