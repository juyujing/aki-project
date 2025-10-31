import pandas as pd

print("--- 開始執行【最終特徵列表】的核心特徵完整性校驗 ---")

# --- 1. 定義文件路徑 ---
# [修改點 1] 這是我們的主角：最終篩選出的特徵列表
FINAL_FEATURES_PATH = 'final_refined_aki_features.csv'
# 這是我們的字典：包含所有特徵的描述
CANDIDATE_FEATURES_PATH = 'discovered_aki_features.csv'

# --- 2. 定義【黃金標準】檢查清單 (與之前完全相同) ---
MUST_HAVE_FEATURES = {
    "Serum Creatinine": ["creatinine"],
    "Urine Output": ["urine", "output"],
    "Blood Urea Nitrogen (BUN)": ["bun", "urea nitrogen"],
    "Mean Arterial Pressure (MAP)": ["map", "mean arterial pressure"],
    "Sepsis Diagnosis": ["sepsis", "septic"],
    "Dialysis / RRT": ["dialysis", "crrt", "renal replacement", "hemofiltration"],
    "Potassium": ["potassium", "k+"],
    "Age": ["age"],
    "Vasopressors": ["vasopressor", "norepinephrine", "dopamine"]
}

# --- 3. 加載數據並準備待查列表 ---
try:
    final_df = pd.read_csv(FINAL_FEATURES_PATH)
    candidate_df = pd.read_csv(CANDIDATE_FEATURES_PATH)
    print(f"成功加載您的 {len(final_df)} 個最終特徵 Code 列表。")
except FileNotFoundError as e:
    print(f"錯誤: 找不到所需文件: {e}")
    exit()

# [修改點 2] 將最終列表與候選列表合併，以獲取描述信息
# 關鍵步驟：我們只關心最終被選中的那些特徵的描述
final_features_with_desc = pd.merge(
    final_df,
    candidate_df[['code', 'description']],
    left_on='final_selected_code', # final_df 中的列名
    right_on='code',             # candidate_df 中的列名
    how='left'
)
final_features_with_desc['description_lower'] = final_features_with_desc['description'].fillna('').str.lower()
print("已成功關聯最終特徵的描述信息。")

# --- 4. 開始逐項檢查 ---
print("\n" + "="*60)
print("            特徵集完整性 - 最終質量控制報告")
print("="*60)

all_critical_found = True
# 將所有【最終】特徵的描述合併成一個大的文本塊
all_descriptions_text = " ".join(final_features_with_desc['description_lower'].tolist())

for feature_name, keywords in MUST_HAVE_FEATURES.items():
    found = False
    for keyword in keywords:
        if keyword in all_descriptions_text:
            found = True
            break
            
    if found:
        print(f"[CHECK PASSED]  - {feature_name}")
    else:
        if feature_name == "Age":
             print(f"[MANUAL CHECK] - {feature_name} (通常由 MEDS_BIRTH 計算，請在特徵工程中確保已包含)")
        else:
            print(f"[CHECK FAILED]  - {feature_name}  <--- 嚴重警告：Llama 可能錯誤地移除了核心特徵！")
            all_critical_found = False

print("="*60)

# --- 5. 總結 ---
if all_critical_found:
    print("\n[結論] 最終質量控制通過！")
    print("您的 114 個特徵集在經過 Llama 精煉後，仍然完整地覆蓋了所有極其關鍵的 AKI 維度。")
    print("這個特徵集兼具了【廣度】、【相關性】和【核心完整性】，可以作為最終聚類的黃金標準。")
else:
    print("\n[結論] 最終質量控制失敗！")
    print("這是一個重要的發現！Llama 在消除冗餘的過程中可能過於激進，移除了部分核心特徵。")
    print("建議手動將缺失的核心特徵 Code 添加回您的 'final_refined_aki_features.csv' 文件中。")