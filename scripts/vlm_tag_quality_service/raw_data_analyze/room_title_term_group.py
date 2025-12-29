"""
## room title 分群 (V3 - 針對客餐廳優化)
- Data: ALL vlm_rematch_twhg room title
- Logic Rules:
  1. 優先排除衛浴。
  2. 排除公設/外部/特定非居住功能 (如: 交誼廳, 梯廳, 店面)。
  3. 捕捉 "廳" 類別 (包含: 客廳, 餐廳, 廚房, 客餐廳, 吧台)。
  4. 捕捉 "房" 類別。
  5. 剩下歸類為 "其他" (如: 單獨的玄關, 陽台, 走道)。
"""

import json
import os
import re
from collections import Counter, defaultdict

# 設定您的資料路徑
raw_data_folder_path: str = "../../../data/vlm_rematch_twhg_with_latlng_and_places/"


def clean_room_title(title: str) -> str:
    if not title:
        return ""
    # 移除括號、英數字、特殊符號
    cleaned = re.sub(r'[\(（].*?[\)）]', '', title)
    cleaned = re.sub(r'[A-Za-z0-9]', '', cleaned)
    cleaned = re.sub(r'[/\+\-_]', '', cleaned)
    return cleaned.strip()


def classify_category(title: str) -> str:
    """
    分類邏輯核心
    """

    # --- 1. 衛 (最明確，優先排除) ---
    bath_keywords = ['衛', '浴', '廁', '洗手']
    if any(k in title for k in bath_keywords):
        return '衛(廁所/浴室)'

    # --- 2. 排除非居住區域/公設 (優先於 "廳" 的判斷) ---
    # 這裡要小心，不能誤殺 "客餐廳"
    # "梯廳" 屬於公設或過道，不屬於室內廳
    # "交誼廳" 屬於公設
    public_or_external_keywords = [
        '交誼', '健身', '遊戲', '撞球', '公設',
        '大廳', '門廳', '梯廳', '櫃台', '信箱', '中庭',
        '店面', '騎樓', '商', '辦公',
        '車', '停',
        '頂樓', '外觀', '大門', '花園', '入口', '外牆',
        '電箱', '水塔', '機房', '垃圾'
    ]
    if any(k in title for k in public_or_external_keywords):
        return '其他(公設/車位/外部/店面)'

    # --- 3. 廳 (居住生活的公共區域) ---
    # 包含: 客餐廳, 客廳, 餐廳, 廚房
    # 只要命中這裡，就算是 "客餐廳玄關" 也會歸類在此 (符合主要功能優先原則)
    hall_keywords = [
        '客廳', '客餐廳', '餐廳',  # 完整詞優先
        '廚', '起居',  # 功能詞
        '餐', '吧', '中島',  # 餐廳相關
        # 注意: 不單獨使用 '廳' 字，避免誤判奇怪的複合詞，但 '客' 與 '餐' 已足夠涵蓋
    ]
    # 額外補強: 如果包含 "廳" 且不包含 "梯" 等負面詞 (雖然上面已排除梯廳，但雙重保險)
    if any(k in title for k in hall_keywords) or ('廳' in title and '梯' not in title):
        return '廳(客廳/餐廳/廚房)'

    # --- 4. 房 (居住區域) ---
    room_keywords = ['臥', '房', '書房', '和室', '孝親', '更衣室']
    # 更衣室有時算其他，但通常依附於房間，視需求而定，這裡暫歸房或可移至其他
    if any(k in title for k in room_keywords):
        return '房(書房/臥室/客房)'

    # --- 5. 室內其他 (附屬空間) ---
    # 單獨的 "玄關", "陽台", "走道" 會落到這裡
    indoor_misc_keywords = [
        '玄關', '鞋櫃',
        '走廊', '走道', '梯', '通道',
        '陽台', '露台', '曬衣',
        '儲藏', '倉', '置物'
    ]
    if any(k in title for k in indoor_misc_keywords):
        return '其他(玄關/陽台/走道/儲藏)'

    # --- 6. 無法識別 ---
    return '未分類/其他'


def process_files(folder_path):
    json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    category_stats = defaultdict(Counter)

    # 測試用：追蹤特定詞彙的去向
    test_cases = ['客餐廳', '客餐廳玄關', '玄關', '交誼廳', '梯廳', '餐吧區']
    test_logs = []

    print(f"開始處理 {len(json_files)} 個 JSON 檔案...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            result_list = data.get("matching_result_list", [])

            for item in result_list:
                extracted = item.get("extracted_feature", {})
                raw_room = extracted.get("room", "")

                if not raw_room:
                    continue

                clean_room = clean_room_title(raw_room)
                if not clean_room:
                    continue

                category = classify_category(clean_room)
                category_stats[category][clean_room] += 1

                # 記錄我們關心的測試案例
                for case in test_cases:
                    if case in clean_room:
                        test_logs.append((clean_room, category))
                        break

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return category_stats, test_logs


# 執行
stats, logs = process_files(raw_data_folder_path)

# 輸出結果
print("\n" + "=" * 50)
print("【各分類統計結果】")
print("=" * 50)

sort_order = [
    '廳(客廳/餐廳/廚房)',
    '房(書房/臥室/客房)',
    '衛(廁所/浴室)',
    '其他(玄關/陽台/走道/儲藏)',
    '其他(公設/車位/外部/店面)',
    '未分類/其他'
]

for category in sort_order:
    if category in stats:
        word_counter = stats[category]
        print(f"\n📂 分類: {category}")
        print(f"   總計數量: {sum(word_counter.values())}")
        print(f"   詞頻分佈 (Top 5):")
        for word, count in word_counter.most_common():
            print(f"     - {word}: {count}")

print("\n" + "=" * 50)
print("【邏輯驗證 (關鍵詞檢查)】")
print("=" * 50)
# 去重後顯示測試案例的分類結果
unique_logs = sorted(list(set(logs)), key=lambda x: x[1])
for room, cat in unique_logs:
    if any(k in room for k in ['客餐廳', '玄關', '交誼', '梯廳']):
        print(f"詞彙: {room:<15} -> {cat}")
