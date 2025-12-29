import asyncio
import json
from pathlib import Path

from services.vlm_tag_quality_service import using_vlm_with_spatial_signals_info_as_a_judge


async def process_single_file_sequential(input_file_path: Path, output_dir: Path):
    """
    循序處理單一檔案內的每一個 Item，避免顯存爆掉。
    """
    print(f"📂 [開始檔案] {input_file_path.name}")

    try:
        # 1. 讀取檔案
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        items_to_process = data.get("matching_result_list", [])

        if not items_to_process:
            print(f"   ⚠️ [跳過] {input_file_path.name} (無資料)")
            return

        output_data = []
        total_items = len(items_to_process)

        # 2. 【關鍵修改】使用 for loop 配合 await 進行循序處理
        # 不使用 asyncio.gather，確保一次只對 Model 發送一個請求
        for index, item in enumerate(items_to_process):
            extracted_feature = item.get('extracted_feature', {})

            # 提取參數
            room_name = extracted_feature.get('room')
            tag_info = extracted_feature.get('tag_list', [])
            raw_description = extracted_feature.get('raw_description')
            image_url = item.get('matched_image')

            print(f"   > 正在處理 Item {index + 1}/{total_items} ...", end="\r")

            try:
                # 這裡直接 await，程式會暫停直到這張圖跑完，才跑下一張
                response = await using_vlm_with_spatial_signals_info_as_a_judge(
                    room_name=room_name,
                    tag_info=" ".join(tag_info) if isinstance(tag_info, list) else str(tag_info),
                    raw_description=raw_description,
                    image_url=image_url
                )

                # 處理 Pydantic output (如果 response 是 Pydantic model)
                if hasattr(response, 'model_dump'):
                    judge_result = response.model_dump()
                else:
                    judge_result = response

            except Exception as e:
                # 捕捉單一圖片處理失敗，不影響整個檔案
                print(f"\n   ❌ Item {index + 1} 失敗: {e}")
                judge_result = {"error": str(e), "status": "failed"}

            # 3. 組合結果 (與 Script 2 格式保持一致)
            output_entry = {
                "original_input": item,  # 保留原始輸入以利 Trace
                "evaluation": judge_result
            }
            output_data.append(output_entry)

        # 4. 該檔案全部跑完後，寫入結果
        output_file_path = output_dir / input_file_path.name
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"\n✅ [完成檔案] {input_file_path.name} -> 已存檔")

    except Exception as e:
        print(f"\n⛔ [檔案錯誤] {input_file_path.name} 發生異常: {e}")


async def main():
    # 設定路徑
    input_dir = Path("../../data/vlm_rematch_add_info_twhg_with_latlng_and_places/")
    # 輸出路徑設定為你指定的 vlm_as_a_judge
    output_dir = Path("../../data/vlm_tag_quality_service/vlm_with_spatial_signals_info_as_a_judge/")

    # 建立目錄
    output_dir.mkdir(parents=True, exist_ok=True)

    # 取得所有 .json 檔案並排序
    json_files = sorted(list(input_dir.glob("*.json")))
    total_files = len(json_files)

    print(f"🚀 總共發現 {total_files} 個檔案，將採循序模式處理 (Save VRAM Mode)...")

    # 逐一處理每個檔案
    for i, file_path in enumerate(json_files):
        print(f"\n--- 進度: 檔案 {i + 1} / {total_files} ---")
        # 直接 await，確保一個檔案處理完才換下一個
        await process_single_file_sequential(file_path, output_dir)

    print("\n🎉 所有檔案處理完成！")


if __name__ == "__main__":
    asyncio.run(main())