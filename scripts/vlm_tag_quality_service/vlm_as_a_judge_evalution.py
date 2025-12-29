import asyncio
import json
from pathlib import Path

from services.vlm_tag_quality_service import using_vlm_as_a_judge


async def process_single_file(input_file_path: Path, output_dir: Path):
    """
    處理單一 JSON 檔案的邏輯
    """
    print(f"  [開始] 處理檔案: {input_file_path.name}")

    try:
        # 1. 讀取檔案
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tasks = []
        items_to_process = data.get("matching_result_list", [])

        if not items_to_process:
            print(f"  [跳過] {input_file_path.name} (無資料)")
            return

        # 2. 建立該檔案內所有圖片的任務
        for image_object in items_to_process:
            feature = image_object["extracted_feature"]
            task = using_vlm_as_a_judge(
                room_name=feature["room"],
                tag_info=" ".join(feature["tag_list"]),
                raw_description=feature["raw_description"],
                image_url=image_object["matched_image"]
            )
            tasks.append(task)

        # 3. 並發處理該檔案內的所有圖片
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. 整理結果 (這裡是用來修復 JSON serializable error 的關鍵)
        output_data = []
        for original_obj, result in zip(items_to_process, results):

            # --- 錯誤處理 ---
            if isinstance(result, Exception):
                # 如果是連線錯誤或其他 Exception，轉成字串存起來
                judge_result = {"error": str(result), "status": "failed"}

            # --- Pydantic 物件處理 (關鍵修改) ---
            else:
                # 判斷是否為 Pydantic 物件並轉為 Dict
                # Pydantic v2 使用 .model_dump()
                judge_result = result.model_dump()

            # --- 組合資料以確保 Traceability ---
            output_entry = {
                # 這裡保留原始的所有資訊 (包含 Image URL, Tags, Room Name)
                # 讓你日後可以 Trace 回去
                "original_input": original_obj,

                # 這裡是 VLM 的評分結果
                "evaluation": judge_result
            }
            output_data.append(output_entry)

        # 5. 寫入檔案
        output_file_path = output_dir / input_file_path.name
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 讓中文可以正常顯示，不會變成 \uXXXX
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"  [完成] {input_file_path.name} -> 已存檔")

    except Exception as e:
        print(f"  [錯誤] 檔案 {input_file_path.name} 發生異常: {e}")


async def main():
    # 設定路徑
    input_dir = Path("../../data/vlm_rematch_add_info_twhg_with_latlng_and_places/")
    output_dir = Path("../../data/vlm_tag_quality_service/vlm_as_a_judge/")

    # 建立目錄 (包含 parents)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 取得所有 .json 檔案並排序
    json_files = sorted(list(input_dir.glob("*.json")))
    total_files = len(json_files)

    # 設定 Batch Size (一次處理幾個 JSON 檔)
    BATCH_SIZE = 20

    print(f"總共發現 {total_files} 個檔案，每次處理 {BATCH_SIZE} 個...")

    # 使用 range 與 step 來進行 Batch 切分
    for i in range(0, total_files, BATCH_SIZE):
        # 取出目前的 batch
        current_batch_files = json_files[i: i + BATCH_SIZE]

        print(f"\n=== 正在執行 Batch {i // BATCH_SIZE + 1} / {(total_files + BATCH_SIZE - 1) // BATCH_SIZE} ===")
        print(f"目標檔案: {[f.name for f in current_batch_files]}")

        # 建立這個 Batch 的檔案處理任務清單
        file_tasks = [process_single_file(f, output_dir) for f in current_batch_files]

        # 等待這個 Batch 的所有檔案都處理完，才進入下一個 Batch
        await asyncio.gather(*file_tasks)

    print("\n所有 Batch 處理完成！")


if __name__ == "__main__":
    asyncio.run(main())
