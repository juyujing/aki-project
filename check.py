import pandas as pd
import numpy as np

print("--- 開始執行數據驅動的 AKI 特徵發現腳本 ---")
print("--- [約束] 本腳本只使用 MEDS cohort 中的 metadata/codes.parquet ---")

# --- 1. 定義文件路徑 ---
MEDS_METADATA_PATH = './my_meds_demo_output/MEDS_cohort/metadata/codes.parquet'

# --- 2. 定義全面的 AKI 相關關鍵詞庫 (按臨床邏輯分類) ---
AKI_KEYWORDS = {
    "CORE_KIDNEY_FUNCTION": [
        'creatinine', 'bun', 'urea nitrogen', 'urine output', 'urine volume', 'urine', 'renal'
    ],
    "HEMODYNAMICS_PERFUSION": [
        'pressure', 'map', 'sbp', 'dbp', 'cardiac output', 'shock', 'lactate', 'lactic acid',
        'heart rate'
    ],
    "ELECTROLYTES_ACID_BASE": [
        'potassium', 'k+', 'sodium', 'na+', 'bicarbonate', 'hco3', 'chloride', 'anion gap',
        'ph', 'base excess'
    ],
    "HEMATOLOGY_INFLAMMATION": [
        'platelet', 'plt', 'white blood cell', 'wbc', 'hemoglobin', 'hgb', 'hematocrit',
        'neutrophil', 'lymphocyte', 'crp', 'c-reactive protein', 'procalcitonin'
    ],
    "COMORBIDITIES_CAUSES": [
        'sepsis', 'septic', 'diabetes', 'diabetic', 'hypertension', 'htn', 'heart failure',
        'liver disease', 'cirrhosis', 'ckd', 'chronic kidney'
    ],
    "TREATMENTS_INTERVENTIONS": [
        'dialysis', 'crrt', 'hemofiltration', 'renal replacement', 'vasopressor',
        'norepinephrine', 'dopamine', 'epinephrine', 'vasopressin', 'phenylephrine',
        'diuretic', 'furosemide', 'lasix', 'ventilator', 'ventilation', 'intubated'
    ]
}

# --- 3. 加載 MEDS 特徵字典 ---
try:
    print(f"\n正在加載 MEDS 特徵字典: {MEDS_METADATA_PATH}")
    codes_df = pd.read_parquet(MEDS_METADATA_PATH)
    # 關鍵預處理：將 description 列中的空值替換為空字符串，並全部轉為小寫
    codes_df['description_lower'] = codes_df['description'].fillna('').str.lower()
    print(f"字典加載成功，共包含 {len(codes_df)} 個特徵條目。")
except FileNotFoundError:
    print(f"錯誤: 特徵字典文件未找到，請檢查路徑。")
    exit()

# --- 4. 進行大規模關鍵詞匹配 ---
print("\n正在遍歷字典並匹配 AKI 相關關鍵詞...")
found_features = {}

# 為每個類別初始化一個空列表
for category in AKI_KEYWORDS:
    found_features[category] = []

# 遍歷字典的每一行
for index, row in codes_df.iterrows():
    desc = row['description_lower']
    if not desc:
        continue # 如果描述為空，則跳過

    # 遍歷所有關鍵詞類別
    for category, keywords in AKI_KEYWORDS.items():
        # 遍歷該類別下的所有關鍵詞
        for keyword in keywords:
            if keyword in desc:
                # 如果匹配成功，記錄這個特徵的完整信息
                feature_info = {
                    'code': row['code'],
                    'description': row['description'],
                    'matched_keyword': keyword
                }
                found_features[category].append(feature_info)
                # 找到一個匹配就跳出內層循環，避免重複添加
                break 
print("匹配完成。")

# --- 5. 打印詳盡的、分類的特徵報告 ---
print("\n\n" + "="*80)
print("                       大規模 AKI 相關特徵發現報告")
print("                  (來源: metadata/codes.parquet)")
print("="*80)

total_found = 0
for category, features in found_features.items():
    print("\n" + "-"*30 + f" {category} (找到 {len(features)} 個) " + "-"*30)
    if not features:
        print("在此類別下未找到相關特徵。")
    else:
        # 創建一個 DataFrame 以便美觀地打印
        category_df = pd.DataFrame(features)
        with pd.option_context('display.max_rows', None, 'display.max_colwidth', 80):
            print(category_df.to_string(index=False))
    total_found += len(features)

print("\n" + "="*80)
print(f"報告總結：共發現 {total_found} 個與 AKI 預測可能相關的特徵。")
print("="*80)

# --- 6. (可選) 保存結果到 CSV ---
all_features_list = []
for category, features in found_features.items():
    for feature in features:
        feature['category'] = category
        all_features_list.append(feature)

final_df = pd.DataFrame(all_features_list)
final_df.to_csv('discovered_aki_features.csv', index=False)
print("\n[INFO] 完整的特徵列表已保存到 discovered_aki_features.csv 文件中。")