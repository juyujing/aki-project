import pandas as pd
import numpy as np # 仍然需要 numpy 来确保代码完整性

print("--- 開始執行【最終修正版：類型約束】的完整性校驗與修正腳本 ---")

# --- 1. 定義文件路徑 (與您提供的一致) ---
LLAMA_FINAL_LIST_PATH = 'final_refined_aki_features.csv'
CANDIDATE_FEATURES_PATH = 'discovered_aki_features.csv'
CORRECTED_FINAL_LIST_PATH = 'aki_features.csv' # 输出文件名保持一致

# --- 2. [核心修正] 定義【黃金標準】檢查清單 (加入強制類型約束) ---
GOLD_STANDARD_CONCEPTS_WITH_TYPE = {
    # 核心數值型特徵 (LAB 或 CHARTEVENTS)
    "Serum Creatinine": (["LAB"], ["creatinine"]),
    "Blood Urea Nitrogen (BUN)": (["LAB"], ["bun", "urea nitrogen"]),
    "Potassium": (["LAB"], ["potassium", "k+"]),
    
    # 核心生命體徵 (CHARTEVENTS)
    "Mean Arterial Pressure (MAP)": (["CHARTEVENTS"], ["map", "mean arterial pressure"]),
    
    # 核心輸出/干預 (OUTPUTEVENTS/PROCEDUREEVENTS)
    "Urine Output": (["OUTPUTEVENTS"], ["urine", "output"]),
    "Dialysis / RRT": (["PROCEDUREEVENTS"], ["dialysis", "crrt", "renal replacement", "hemofiltration"]),
    "Vasopressors": (["INPUTEVENTS"], ["vasopressor", "norepinephrine", "dopamine"]),
    
    # 核心診斷/靜態特徵
    "Sepsis Diagnosis": (["DIAGNOSIS"], ["sepsis", "septic"]),
    "Age": (["STATIC"], ["age"]),
}


# --- 3. 加載數據 ---
try:
    llama_df = pd.read_csv(LLAMA_FINAL_LIST_PATH)
    candidate_df = pd.read_csv(CANDIDATE_FEATURES_PATH)
    print(f"成功加載 Llama 的 {len(llama_df)} 個特徵和 {len(candidate_df)} 個候選特徵。")
except FileNotFoundError as e:
    print(f"錯誤: 找不到所需文件: {e}")
    exit()

# --- 4. 類型約束的校驗與添加 ---
print("\n" + "="*60)
print("           對 Llama 的篩選結果進行最終校驗與修正 (類型約束)")
print("="*60)

final_codes_set = set(llama_df['final_selected_code'])
missing_codes_to_add = set()
candidate_df['description_lower'] = candidate_df['description'].fillna('').str.lower()


# 準備 Llama 選擇的特徵及其描述，用於後續檢查
final_features_with_desc = pd.merge(
    llama_df, candidate_df[['code', 'description']],
    left_on='final_selected_code', right_on='code', how='left'
)
final_features_with_desc['description_lower'] = final_features_with_desc['description'].fillna('').str.lower()

# 遍歷必備標準清單
for concept_name, (allowed_types, keywords) in GOLD_STANDARD_CONCEPTS_WITH_TYPE.items():
    is_present_in_llama_list = False
    
    # 遍歷 Llama 選擇的每一個特徵的描述，檢查是否匹配
    for keyword in keywords:
        # 檢查 Llama 選擇的列表中，是否有特徵匹配關鍵詞
        if final_features_with_desc['description_lower'].str.contains(keyword, regex=False).any():
            is_present_in_llama_list = True
            break
    
    # 在檢查完所有關鍵詞後，做出判斷
    if is_present_in_llama_list:
        print(f"[FOUND]     - {concept_name} (概念已存在)。")
    else:
        # --- 檢索並添加的邏輯 ---
        print(f"[MISSING]   - {concept_name} 在 Llama 的選擇中缺失，需要強制添加。")
        found_in_candidate = False
        
        # 遍歷允許的類型
        for type_prefix in allowed_types:
            if found_in_candidate: break

            # 1. 篩選出只屬於該類型 (e.g., 'LAB') 的候選特徵
            type_filtered_candidates = candidate_df[candidate_df['code'].str.startswith(type_prefix, na=False)]
            
            for keyword in keywords:
                # 2. 在該類型中搜索關鍵詞
                matches = type_filtered_candidates[type_filtered_candidates['description_lower'].str.contains(keyword, case=False, na=False, regex=False)]
                
                if not matches.empty:
                    # 找到了！獲取第一個匹配項的精確 Code
                    code_to_add = matches['code'].iloc[0]
                    
                    print(f"              -> [ADDING] 已從 '{type_prefix}' 類中檢索到 Code: {code_to_add} 並將其強制加入。")
                    missing_codes_to_add.add(code_to_add)
                    found_in_candidate = True
                    break 

        if not found_in_candidate and concept_name != "Age":
             print(f"              -> [WARNING] 在候選列表中也未找到 '{concept_name}'，請檢查關鍵詞。")


# --- 5. 合併與保存 ---
print("="*60)
print("\n正在合併 Llama 的選擇與強制添加的核心特徵...")
corrected_codes_set = final_codes_set.union(missing_codes_to_add)
print(f"  - Llama 原始選擇數量: {len(final_codes_set)}")
print(f"  - 強制添加的缺失數量: {len(missing_codes_to_add)}")
print(f"  - 修正後最終特徵總數: {len(corrected_codes_set)}")
final_corrected_df = pd.DataFrame(list(corrected_codes_set), columns=['final_selected_code'])
final_corrected_df.to_csv(CORRECTED_FINAL_LIST_PATH, index=False)

print(f"\n修正完成！包含 {len(final_corrected_df)} 個特徵的特徵集已生成。")
print(f"結果已保存到: {CORRECTED_FINAL_LIST_PATH}")

# 被添加的 Code 列表
print("\n--------------------------------------------------------------------------------")
print("被程序強制添加的核心 CODE 列表:")
for code in missing_codes_to_add:
    # 這裡我們需要打印出 Code 的描述
    desc = candidate_df[candidate_df['code'] == code]['description'].iloc[0]
    print(f"   - {code}: {desc}")
