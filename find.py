import pandas as pd

print("--- 開始執行【最終特徵列表】核心 Code 真相查找腳本 ---")

# --- 1. 定義文件路徑 ---
FINAL_FEATURES_PATH = 'aki_features.csv'
CANDIDATE_FEATURES_PATH = 'discovered_aki_features.csv' # 需要這個文件來獲取描述

# --- 2. 定義我們需要查找的核心臨床概念及其關鍵詞 ---
# 這些是我們知道必須存在的東西
CORE_CONCEPTS_KEYWORDS = {
    'BUN': ['bun', 'urea nitrogen'],
    'Creatinine': ['creatinine'],
    'Lactate': ['lactate', 'lactic acid'],
}

# --- 3. 加載數據並合併描述 ---
try:
    final_df = pd.read_csv(FINAL_FEATURES_PATH)
    candidate_df = pd.read_csv(CANDIDATE_FEATURES_PATH)
    
    # [核心] 將 final_df 與描述合併
    final_features_with_desc = pd.merge(
        final_df,
        candidate_df[['code', 'description']],
        left_on=final_df.columns[0], # 使用第一列作為 code
        right_on='code',
        how='left'
    )
    final_features_with_desc['description_lower'] = final_features_with_desc['description'].fillna('').str.lower()
    
    # 確保最終特徵列表的 Code 仍然在第一列
    final_features_with_desc = final_features_with_desc.rename(columns={final_df.columns[0]: 'final_code'})

    print(f"成功加載 {len(final_df)} 個 Code 並關聯描述。")

except FileNotFoundError as e:
    print(f"錯誤: 找不到所需文件: {e}")
    exit()

# --- 4. 執行文本匹配查找 ---
found_codes_map = {}
for concept, keywords in CORE_CONCEPTS_KEYWORDS.items():
    
    # 遍歷所有關鍵詞，在描述列中查找匹配的行
    matching_rows = pd.DataFrame()
    for keyword in keywords:
        # 使用 .str.contains 進行文本匹配
        matches = final_features_with_desc[final_features_with_desc['description_lower'].str.contains(keyword, na=False, regex=False)]
        matching_rows = pd.concat([matching_rows, matches]).drop_duplicates(subset=['final_code'])
    
    if not matching_rows.empty:
        # 找到了！選擇第一個匹配項的精確 Code 作為黃金標準
        gold_code = matching_rows['final_code'].iloc[0]
        found_codes_map[concept] = gold_code
    else:
        found_codes_map[concept] = None

# --- 5. 打印最終報告 ---
print("\n" + "="*80)
print("              AKI 核心特徵 Code (來自 aki_features.csv) 報告")
print("="*80)

final_codes_for_correction = {}
all_ok = True

for concept in ['BUN', 'Creatinine', 'Lactate']:
    code = found_codes_map.get(concept)
    if code:
        print(f"[FOUND] - {concept}: {code}")
        final_codes_for_correction[concept] = code
    else:
        print(f"[MISSING] - {concept}！無法在最終列表中找到該概念的核心 Code。")
        all_ok = False

print("="*80)

if not all_ok:
    print("\n[結論] 存在致命的邏輯錯誤！請返回上一步，強制將缺失的特徵注入到 aki_features.csv 中。")
else:
    print("\n[下一步] 請使用以下字典來修正您的 clustering.py 腳本：")
    print("--------------------------------------------------------------------------------")
    print(f"BUN Code: {final_codes_for_correction.get('BUN')}")
    print(f"Creatinine Code: {final_codes_for_correction.get('Creatinine')}")
    print(f"Lactate Code: {final_codes_for_correction.get('Lactate')}")
    print("--------------------------------------------------------------------------------")