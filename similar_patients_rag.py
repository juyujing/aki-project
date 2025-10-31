import json
import time
import os
import torch
import pandas as pd
import numpy as np
import transformers
from sklearn.metrics.pairwise import cosine_similarity

# 0. 驗證並打印緩存路徑
# 文件路徑
FINAL_EMBEDDING_PATH = 'patient_embedding_array.npy'
FINAL_CLUSTER_CSV_PATH = 'patient_clusters_and_features.csv'
MEDS_DATA_PATH = './my_meds_demo_output/MEDS_cohort/data/train/0.parquet'
AKI_FEATURES_PATH = 'aki_features.csv' # 用于查找原始 Code

cache_dir = os.environ.get("HF_HOME")
if cache_dir:
    print(f"[INFO] 檢測到 HF_HOME 環境變量。模型將被下載並緩存到: {cache_dir}")
else:
    print("[WARNING] 未檢測到 HF_HOME 環境變量。模型將被下載到默認位置 ~/.cache/huggingface/")

# 1. 定義模型 ID
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
print(f"\n[INFO] 開始加載模型: {model_id}")
start_time = time.time()

# 2. 創建 Pipeline (這就是實例化 'pipeline' 變量的關鍵代碼)
try:
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16}, # torch_dtype is the standard argument
        device_map="auto",
    )
    end_time = time.time()
    print(f"[SUCCESS] 模型成功加載，耗時 {end_time - start_time:.2f} 秒。")
except Exception as e:
    print(f"模型加載失敗: {e}")
    exit()

# 加載 Embedding 數據和 ID 映射
try:
    print(f"[INFO] 2/3 正在加載 Embedding 數據和 ID 映射...")
    # 加載真實 Embedding 數組
    PATIENTS_EMBEDDING = np.load(FINAL_EMBEDDING_PATH) 
    
    # 獲取真實的 Patient ID 列表 (必須從保存的 CSV 文件中讀取)
    patient_features_and_labels = pd.read_csv(FINAL_CLUSTER_CSV_PATH, index_col=0)
    PATIENT_ID_MAP = patient_features_and_labels.index.values # 這是真實的 subject_id 列表
    
    # [修正點]: 將讀取到的 DataFrame 賦值給 PATIENT_FEATURES_WITH_LABELS 變量
    # 這個全域變量會在 generate_comparative_aki_code 函數中被使用
    PATIENT_FEATURES_WITH_LABELS = patient_features_and_labels 
    
    # 加載原始事件數據和特徵 Code (用於上下文分析)
    ALL_EVENTS_DF = pd.read_parquet(MEDS_DATA_PATH)
    SELECTED_FEATURE_CODES = pd.read_csv(AKI_FEATURES_PATH).iloc[:, 0].tolist()

    print(f"[SUCCESS] 數據加載完成。共 {len(PATIENT_ID_MAP)} 個病人。")
except FileNotFoundError as e:
    print(f"數據加載失敗: {e}")
    exit()

# 預處理 Patient ID Map，確保類型一致
PATIENT_ID_MAP = PATIENT_ID_MAP.astype(int)


def find_top_k_similar_patients(target_patient_id: int, 
                                embeddings: np.ndarray, 
                                patient_id_map: np.ndarray, 
                                k: int = 10) -> list:
    """
    階段一：使用餘弦相似度找到 Top K 個最相似的患者。
    """
    print(f"\n[PHASE 1] 正在為患者 {target_patient_id} 尋找 Top {k} 個相似患者...")
    
    # 1. 根據患者ID找到其在嵌入矩陣中的索引位置
    try:
        target_index = np.where(patient_id_map == target_patient_id)[0][0]
    except IndexError:
        print(f"錯誤: 目標患者 ID {target_patient_id} 未找到。")
        return []

    # 2. 提取目標患者的嵌入向量，並調整其形狀以進行矩陣運算
    target_embedding = embeddings[target_index].reshape(1, -1)
    
    # 3. 計算目標患者與所有患者的餘弦相似度
    # 這會返回一個包含所有相似度分數的陣列
    sim_scores = cosine_similarity(target_embedding, embeddings)[0]
    
    # 4. 對分數進行排序，找到最高的 K 個
    # np.argsort 返回的是索引。[::-1] 將其反轉，實現從大到小排序。
    # [1:k+1] 選擇了從第2個到第k+1個的索引，巧妙地排除了患者自身（相似度總為1，排第一）。
    top_indices = np.argsort(sim_scores)[::-1][1:k+1]
    
    # 5. 將索引轉換回患者ID
    similar_patient_ids = [int(patient_id_map[i]) for i in top_indices]
    
    print(f"成功找到相似患者 ID: {similar_patient_ids}")
    return similar_patient_ids


def generate_comparative_aki_code(target_patient_id: int, 
                                  similar_patient_ids: list, 
                                  pipeline_instance: transformers.Pipeline) -> str:
    """
    階段二：RAG 模式下的代碼生成。Llama 將生成一個能夠回溯統計數據的函數。
    """
    
    # [核心修正] 從全局 DataFrame 中提取所有可用的數值型特徵列名
    all_numeric_cols = [col for col in PATIENT_FEATURES_WITH_LABELS.columns if '_mean' in col or '_std' in col]

    # 1. [RAG 語境] 構建純 Embedding 關聯的 JSON (不含任何統計數值)
    similar_indices = [np.where(PATIENT_ID_MAP == pid)[0][0] for pid in similar_patient_ids]
    similar_embeddings_list = PATIENTS_EMBEDDING[similar_indices].tolist()

    rag_context = {
        "metadata": {
            "similarity_source": "Autencoder Embedding Space (191 AKI Features)",
            "similar_patient_ids": similar_patient_ids,
            "similar_embeddings_full_list": similar_embeddings_list 
        },
        "data_dependency_info": { 
            "required_df_variables": ["PATIENT_FEATURES_WITH_LABELS"], 
            "available_mean_columns": [col for col in all_numeric_cols if '_mean' in col], # 傳遞所有 Mean 列名
            "available_std_columns": [col for col in all_numeric_cols if '_std' in col],  # 傳遞所有 Std 列名
            "risk_time_points": ["24h", "48h", "72h"]
        },
        "summary": "Target patient is highly similar to this cohort. The generated code MUST calculate dynamic risk thresholds for 24h, 48h, and 72h AKI based on combined cohort statistics."
    }
    
    system_prompt = (
        "You are a sophisticated Python code generation engine. Your task is to generate a single, complete Python function. "
        "The generated function MUST perform the following RAG process: 1) Access the GLOBAL DataFrame 'PATIENT_FEATURES_WITH_LABELS' to perform statistical analysis on the similar cohort's data. 2) Derive time-specific thresholds (24h/48h/72h) from this data by using a COMBINED RISK SCORE model based on the available mean and standard deviation columns. 3) Compare the target patient's data against these thresholds. "
        "The function MUST be runnable without any external file access."
        "Output ONLY the Python code block, starting with ```python and ending with ```. Do not include any other text or explanations."
    )
    
    user_prompt = f"""
    Generate a Python function named `predict_aki_risk_for_{target_patient_id}`.

    **Function Requirements:**
    1.  **Signature**: It MUST accept two parameters: `target_patient_id: int` and `rag_context_json: str`. (It must access PATIENT_FEATURES_WITH_LABELS globally).
    2.  **Logic**: 
        a. **Data Access**: Filter the GLOBAL DataFrame 'PATIENT_FEATURES_WITH_LABELS' using the 'similar_patient_ids' from the context.
        b. **Threshold Calculation (Comprehensive)**: The code MUST calculate a weighted **Cohort Risk Score Mean** using **AT LEAST THREE** of the available mean columns (e.g., Creatinine Mean, BUN Mean, MAP Mean, Lactate Mean) found in the context. Then, define three risk thresholds (24h, 48h, 72h) based on the calculated Cohort Risk Score Mean and Standard Deviation (e.g., Threshold_48h = Mean + 2*StdDev).
        c. **Comparison**: The code must calculate a **Target Risk Score** based on the target patient's data (e.g., Target Risk Score = (Target Creatinine * 0.4) + (Target BUN * 0.2) + ...). Compare the Target Risk Score against the three time-specific thresholds.
    3.  **Return**: Return a dictionary containing the risk level for all three time points, e.g., `{{'24h': 'Low Risk', '48h': 'Moderate Risk', '72h': 'High Risk'}}`.

    **Here is the RAG context (Embedding Association and Data Guide):**
    ```json
    {json.dumps(rag_context, indent=2)}
    ```

    Generate the Python function now.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    print("\n[PHASE 2] 正在向 Llama 模型發送請求以生成 RAG 代碼...")
    start_time = time.time()
    
    try:
        # --- 嚴格仿照您示例的 Llama 模型調用方式 ---
        outputs = pipeline_instance(
            messages,
            max_new_tokens=1024,
            do_sample=False, # 確保輸出穩定
            eos_token_id=pipeline_instance.tokenizer.eos_token_id
        )
        
        end_time = time.time()
        print(f"[SUCCESS] 模型成功生成代碼，耗時 {end_time - start_time:.2f} 秒。")

        # 從模型的返回結果中解析出生成的代碼
        generated_text = outputs[0]["generated_text"][-1]['content']
        
        # 清理潛在的 markdown 標記，以獲取純代碼
        if "```python" in generated_text:
            generated_code = generated_text.split("```python\n")[1].split("```")[0]
        else:
            generated_code = generated_text

        return generated_code

    except Exception as e:
        end_time = time.time()
        print(f"❌ 在調用 Llama 模型時發生錯誤，耗時 {end_time - start_time:.2f} 秒: {e}")
        error_function = (
            f"def predict_aki_risk_for_{target_patient_id}(target_patient_id: int, target_patient_data: dict, rag_context_json: str) -> dict:\n"
            "    # Llama call failed\n"
            f"   return {{'Error': 'Model generation failed due to: {e}'}}"
        )
        return error_function
    
# [最小化修改 1] 將所有調用邏輯移入這個新的主函數
def run_prediction_pipeline(target_patient_id: int):
    
    # 1. 調用第一步的函數
    similar_ids = find_top_k_similar_patients(
        target_patient_id=target_patient_id,
        embeddings=PATIENTS_EMBEDDING,
        patient_id_map=PATIENT_ID_MAP,
        k=10
    )
    
    if similar_ids:
        # 2. [RAG] 調用 Llama 代碼生成函數
        generated_code_str = generate_comparative_aki_code(
            target_patient_id=target_patient_id,
            similar_patient_ids=similar_ids,
            pipeline_instance=pipeline
        )
        
        print("\n" + "="*30 + " Llama 生成的動態比較代碼 " + "="*30)
        print(generated_code_str)
        print("="*92)

        # 3. (可選) 保存生成的代碼到文件
        output_filename = f"prediction_code_{target_patient_id}.py"
        with open(output_filename, 'w') as f:
            f.write(generated_code_str)
            
        print(f"\n[OUTPUT] 最終預測代碼已保存到 {output_filename}")


if __name__ == "__main__":
    
    # 從 CSV 中讀取所有病人 ID
    all_patient_ids = PATIENT_ID_MAP.tolist()
    
    # 循環遍歷所有病人，這裡我們只循環前 5 個病人作為示例
    num_patients_to_process = len(all_patient_ids) # 處理所有病人

    for i in range(num_patients_to_process):
        target_id = all_patient_ids[i]
        
        # 調用主函數生成代碼
        run_prediction_pipeline(target_patient_id=target_id)
        
        # 為了演示和調試，我們只處理前 5 個
        if i >= 4: 
            print("\n[INFO] 演示完成，已處理前 5 位患者。")
            break

'''
Example Code:

import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple

def predict_aki_risk_for_10025(target_patient_id: int, rag_context_json: str) -> Dict[str, str]:
    
    # 1. 解析 RAG 上下文並定義加權參數
    try:
        context = json.loads(rag_context_json)
        similar_ids = context['metadata']['similar_patient_ids']
        
        # 根據 context['data_dependency_info'] 中可用的 mean/std 列，選擇特徵並賦予權重
        # 選擇 Creatinine, BUN, MAP 三個核心指標進行加權
        SELECTED_MEAN_COLS = ['creatinine_mean', 'bun_mean', 'map_mean'] 
        RISK_WEIGHTS = {'creatinine_mean': 0.5, 'bun_mean': 0.3, 'map_mean': 0.2} 

    except (json.JSONDecodeError, KeyError) as e:
        return {'Error': f'Context parsing error: {e}'}

    # 2. 數據訪問與過濾：從全局數據中提取相似患者的數據
    global PATIENT_FEATURES_WITH_LABELS 
    
    cohort_df = PATIENT_FEATURES_WITH_LABELS.loc[PATIENT_FEATURES_WITH_LABELS.index.intersection(similar_ids)]
    
    if cohort_df.empty:
        return {'Error': 'No cohort data found in PATIENT_FEATURES_WITH_LABELS.'}

    # 3. 計算 Cohort Risk Score 的統計量
    
    # a. Cohort Mean Risk Score
    cohort_risk_score_mean = 0
    for col, weight in RISK_WEIGHTS.items():
        if col in cohort_df.columns:
             cohort_risk_score_mean += (cohort_df[col].mean() * weight) 
        else:
             # Fallback to the global mean if column is missing in cohort subset
             cohort_risk_score_mean += (PATIENT_FEATURES_WITH_LABELS[col].mean() * weight)
             
    # b. Cohort StdDev (使用 Creatinine Mean 的標準差作為代理風險分散度的估計)
    if 'creatinine_mean' in cohort_df.columns and cohort_df['creatinine_mean'].std() is not np.nan:
        risk_std_dev = cohort_df['creatinine_mean'].std() * 0.5 
    else:
        # 確保在極端情況下 StdDev 有一個合理的默認值
        risk_std_dev = 0.5 

    # 4. 定義動態風險閾值
    RISK_THRESHOLDS = {
        '24h': cohort_risk_score_mean + (0.8 * risk_std_dev), # Moderate threshold
        '48h': cohort_risk_score_mean + (1.6 * risk_std_dev), # High threshold
        '72h': cohort_risk_score_mean + (2.5 * risk_std_dev), # Very High threshold
    }

    # 5. 計算 Target Risk Score
    try:
        target_row = PATIENT_FEATURES_WITH_LABELS.loc[target_patient_id]
    except KeyError:
        return {'Error': f'Target patient ID {target_patient_id} not found in global DataFrame.'}

    target_risk_score = 0
    for col, weight in RISK_WEIGHTS.items():
        target_risk_score += (target_row.get(col, 0) * weight) 

    # 6. 執行比較和評估
    risk_results = {}
    
    # 根據 72h 閾值確定主要風險等級
    if target_risk_score >= RISK_THRESHOLDS['72h']:
        overall_risk = 'High Risk'
    elif target_risk_score >= RISK_THRESHOLDS['48h']:
        overall_risk = 'Moderate Risk'
    else:
        overall_risk = 'Low Risk'

    explanation = (
        f"Target Risk Score ({target_risk_score:.3f}) compared to Cohort Mean ({cohort_risk_score_mean:.3f}). "
        f"72h Threshold: {RISK_THRESHOLDS['72h']:.3f}. Overall Risk: {overall_risk}. "
        f"The prediction relies on the target's weighted mean being higher than the cohort's {overall_risk} threshold."
    )

    # 填充所有時間點的風險結果
    risk_results['24h'] = 'Low Risk' if target_risk_score < RISK_THRESHOLDS['24h'] else 'Moderate Risk'
    risk_results['48h'] = 'Low Risk' if target_risk_score < RISK_THRESHOLDS['48h'] else overall_risk
    risk_results['72h'] = overall_risk
    risk_results['explanation'] = explanation

    return risk_results

'''
