# plot_clusters.py

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

print("\n--- 正在生成聚類結果的可視化圖 ---")

# --- 0. [核心修改] 從 .npy 文件加載數據 ---
# 我們取消下面兩行的註釋，讓腳本從磁盤讀取之前保存的結果。
# 這使得此腳本可以完全獨立於模型訓練腳本來運行。
try:
    print("正在從 .npy 文件加載 Embedding 數組和聚類標籤...")
    embedding_array = np.load('patient_embedding_array.npy')
    cluster_labels = np.load('patient_cluster_labels.npy')
    print(f"數據加載成功！Embedding 形狀: {embedding_array.shape}, 標籤數量: {len(cluster_labels)}")
except FileNotFoundError as e:
    print(f"錯誤: 找不到 .npy 文件: {e}")
    print("請確保 'patient_embedding_array.npy' 和 'patient_cluster_labels.npy' 文件與此腳本在同一目錄下。")
    exit()

# --- 1. 使用 t-SNE 進行降維 ---
# 我們將 32 維的 embedding 降到 2 維
print("正在執行 t-SNE 降維，這可能需要一些時間...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
embedding_2d = tsne.fit_transform(embedding_array)
print("t-SNE 降維完成。")

# --- 2. 使用 Matplotlib 繪製散點圖 ---
plt.figure(figsize=(12, 10))

# 使用 embedding_2d 的第一列作為 x 軸，第二列作為 y 軸
# 關鍵：使用 cluster_labels 作為顏色 (c)
# cmap 是一個顏色映射，讓不同的簇顯示不同的顏色
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, s=50)

# --- 3. 美化圖表 ---
plt.title('AKI Patient Clusters Visualization (t-SNE)', fontsize=16)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 創建一個圖例，標明每個顏色對應哪個簇
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)

# 保存圖像
plt.savefig('aki_patient_clusters.png')
print("可視化圖已保存到: aki_patient_clusters.png")

# 顯示圖像
plt.show()

print("\n--- 腳本執行完畢 ---")