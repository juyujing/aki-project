# run_llama_final_selection.py

import torch
import transformers
import time
import os
import pandas as pd
import json
import re

# --- 0. 準備工作 (與您的代碼相同) ---
cache_dir = os.environ.get("HF_HOME")
if cache_dir:
    print(f"[INFO] HF_HOME 緩存路徑: {cache_dir}")
else:
    print("[WARNING] 未設置 HF_HOME。")

# --- 1. 加載模型 (與您的代碼相同) ---
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
print(f"\n[INFO] 開始加載模型: {model_id} (這一步需要較長時間，請耐心等待...)")
start_time = time.time()
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"dtype": torch.bfloat16},
    device_map="auto",
)
end_time = time.time()
print(f"模型成功加載，耗時 {end_time - start_time:.2f} 秒。")

# --- 2. [修改點] 加載我們的 646 個候選特徵 ---
CANDIDATE_FEATURES_PATH = 'discovered_aki_features.csv'
try:
    candidate_df = pd.read_csv(CANDIDATE_FEATURES_PATH)
    print(f"\n[INFO] 成功加載 {len(candidate_df)} 個候選特徵，準備進行最終篩選。")
except FileNotFoundError:
    print(f"錯誤: 找不到 {CANDIDATE_FEATURES_PATH} 文件。")
    exit()

# --- 3. [修改點] 設計用於「精煉篩選」的 Prompt ---
system_prompt = """
You are an expert clinical data scientist and a nephrologist. 
Your task is to perform a final review on a pre-selected list of candidate features to build a concise, powerful, and non-redundant feature set for predicting Acute Kidney Injury (AKI).
"""

prompt_template = """
I am building a feature set for an AKI prediction model. I have already manually selected a non-negotiable set of **Gold Standard** features: 'Serum Creatinine', 'Urine Output', 'Blood Urea Nitrogen (BUN)', 'Mean Arterial Pressure (MAP)', 'Sepsis Diagnosis', 'Dialysis / RRT', 'Potassium', 'Age', and 'Vasopressors'.

Your task is to review the following **Candidate List** from the "{category_name}" category and select only the **ADDITIONAL** features that provide valuable, complementary information.

Your selection criteria should be:
1.  **High Signal**: Prioritize features that are direct, high-signal indicators of related physiological states (e.g., inflammation, acid-base balance, other organ dysfunction).
2.  **Complementary Value**: Select features that add new, independent dimensions of information not already captured by the Gold Standard set mentioned above.
3.  **Avoid Redundancy**: Do NOT select features that are just minor variations of the Gold Standard concepts I have already selected. For example, do not add another basic creatinine measurement.
4.  **Clinical Relevance**: Ensure every selected feature has a strong, scientifically-backed link to AKI pathophysiology.

Here is the candidate list for this category:
---
{feature_list_str}
---

Your response MUST be a JSON formatted list of strings, where each string is the exact "Code" of a feature you have selected. If you think no additional features are needed from this list, return an empty list [].
Example Response: ["LAB//50813", "LAB//51265"]
"""

# JSON 解析函數
def parse_json_from_llama_response(text: str) -> list:
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match: return None
    json_str = match.group(0)
    try:
        # 處理帶有尾隨逗號的不規範 JSON
        json_str = re.sub(r',\s*\]', ']', json_str)
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

# --- 4. 循環執行每個類別的篩選任務 ---
final_selected_features = []
# 按 'category' 列進行分組
for category_name, group_df in candidate_df.groupby('category'):
    print("\n" + "="*20 + f" 正在為【{category_name}】類別進行精選 " + "="*20)
    
    # 準備該類別的特徵列表字符串
    feature_list_for_prompt = group_df.apply(
        lambda row: f"Code: {row['code']}, Description: {row['description']}",
        axis=1
    ).tolist()
    feature_list_str = "\n".join(feature_list_for_prompt)

    # 構造 Prompt
    user_input = prompt_template.format(category_name=category_name, feature_list_str=feature_list_str)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    print(f"[INFO] 正在為 {len(group_df)} 個候選特徵生成篩選建議...")
    start_time = time.time()
    
    # 執行推理
    outputs = pipeline(
        messages,
        max_new_tokens=2048, # 確保有足夠空間生成列表
        do_sample=False,   # 使用確定性輸出
        pad_token_id=pipeline.tokenizer.eos_token_id,
    )
    
    end_time = time.time()
    
    # 處理並解析模型的回答
    model_response_text = outputs[0]["generated_text"][-1]['content']
    selected_codes_in_category = parse_json_from_llama_response(model_response_text)
    
    if selected_codes_in_category is not None:
        # 清洗 Code
        cleaned_codes = [s.replace("Code: ", "").strip() for s in selected_codes_in_category]
        print(f"Llama 從該類別中精選出 {len(cleaned_codes)} 個特徵，耗時 {end_time - start_time:.2f} 秒。")
        final_selected_features.extend(cleaned_codes)
    else:
        print(f"該類別解析失敗，將保留所有候選特徵以供手動審查。")
        # 作為備份策略，如果解析失敗，我們保留該類別的所有原始候選
        final_selected_features.extend(group_df['code'].tolist())


# --- 5. 保存最終結果 ---
final_df = pd.DataFrame({'final_selected_code': final_selected_features})
# 去重
final_df = final_df.drop_duplicates()

final_df.to_csv('final_refined_aki_features.csv', index=False)

print("\n" + "="*50)
print(f"最終特徵精煉完成！")
print(f"共篩選出 {len(final_df)} 個高度相關且低冗餘的特徵。")
print("結果已保存到 final_refined_aki_features.csv")
print("="*50)