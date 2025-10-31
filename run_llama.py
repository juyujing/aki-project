# run_transformers.py
import torch
import transformers
import time
import os

# --- 0. 檢正緩存路徑 ---
# 從環境變量中獲取緩存路徑，如果未設置則告知用戶
cache_dir = os.environ.get("HF_HOME")
if cache_dir:
    print(f"[INFO] 檢測到 HF_HOME 環境變量。模型將被下載並緩存到: {cache_dir}")
else:
    print("[WARNING] 未檢測到 HF_HOME 環境變量。模型將被下載到默認位置 ~/.cache/huggingface/")


# --- 1. 定義模型 ID ---
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

print(f"\n[INFO] 開始加載模型: {model_id}")
start_time = time.time()

# --- 2. 創建 Pipeline ---
# device_map="auto" 會自動使用您的 B200 GPU
# [關鍵] 此處的 'model' 參數會觸發下載，下載位置遵循 HF_HOME 設置
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

end_time = time.time()
print(f"[SUCCESS] 模型成功加載，耗時 {end_time - start_time:.2f} 秒。")

# --- 3. 準備多輪提問的輸入 ---
# 命名為 input_message 的 list，存儲所有 user 的提問
input_message = [
    "請用不超過100個字解釋一下什麼是黑洞？",
    "如果我掉進去會發生什麼事？",
    "有任何方法可以從中逃脫嗎？",
    "給我一段python代碼來模擬黑洞的引力場。"
]

# 初始化對話歷史，首先放入系統提示
messages = [
    {"role": "system", "content": "You are a helpful assistant providing concise and accurate answers in Chinese."},
]

# --- 4. 循環執行多輪對話 ---
for i, user_input in enumerate(input_message):
    print("\n" + "="*20 + f" 第 {i+1} 輪對話 " + "="*20)
    print(f"使用者提問: {user_input}")

    # 將當前用戶問題添加到對話歷史中
    messages.append({"role": "user", "content": user_input})

    print("\n[INFO] 正在生成回答...")
    start_time = time.time()

    # 執行推理並獲取輸出
    # 將完整的 messages 傳遞給 pipeline
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

    end_time = time.time()
    print(f"[SUCCESS] 回答生成完畢，耗時 {end_time - start_time:.2f} 秒。")

    # 處理並獲取模型的最新回答
    # outputs[0]["generated_text"] 返回的是包含所有對話歷史的完整列表
    model_response = outputs[0]["generated_text"][-1]
    generated_text = model_response['content']

    print(f"\n模型的回答:\n{generated_text}")

    # 關鍵步驟：將模型的回答也添加到對話歷史中，為下一輪對話提供上下文
    messages.append({"role": "assistant", "content": generated_text})

print("\n" + "="*50)
print("[INFO] 所有對話已完成。")
print("="*50)