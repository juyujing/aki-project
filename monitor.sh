#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- 設置 ---
# 強制將日誌文件定義在腳本所在的目錄下，並使用 .txt 擴展名
VRAM_LOG_FILE="$SCRIPT_DIR/vram_usage.txt"
SLEEP_INTERVAL=5

# --- 腳本主體 ---

# 1. 清理舊日誌並啟動後台監控
echo "VRAM 監控已啟動，日誌將強制寫入到 $VRAM_LOG_FILE" > "$VRAM_LOG_FILE"
while true; do
    echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---" >> "$VRAM_LOG_FILE"
    nvidia-smi >> "$VRAM_LOG_FILE"
    sleep $SLEEP_INTERVAL
done &
MONITOR_PID=$!

# 2. 打印提示資訊
echo "[INFO] VRAM 監控已在後台啟動 (PID: $MONITOR_PID)。"
echo "[INFO] 現在開始運行您的 Python 程式..."

# 3. 運行您的 Python 程式
python "$SCRIPT_DIR/run_llama.py"

# 4. 程式結束後，停止監控
echo "[INFO] Python 程式運行完畢。"
kill $MONITOR_PID
echo "[SUCCESS] VRAM 監控已停止。所有任務完成。"
echo "日誌文件已保存在: $VRAM_LOG_FILE"