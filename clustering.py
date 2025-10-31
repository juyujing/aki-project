import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print("--- 開始執行【最終終極解決方案】的無監督聚類流程 ---")

# --- 1 & 2. 加載數據 (不變) ---
MEDS_DATA_PATH = './my_meds_demo_output/MEDS_cohort/data/train/0.parquet'
FINAL_FEATURES_PATH = 'aki_features.csv' 
INVENTORY_PATH = 'feature_inventory.csv' 

try:
    events_df = pd.read_parquet(MEDS_DATA_PATH)
    final_feature_codes_df = pd.read_csv(FINAL_FEATURES_PATH)
    inventory_df = pd.read_csv(INVENTORY_PATH)
    
    selected_codes = final_feature_codes_df[final_feature_codes_df.columns[0]].tolist()
    print(f"成功加載 MEDS 數據和 {len(selected_codes)} 個最終選擇的特徵 Code。")
except FileNotFoundError as e:
    print(f"錯誤: 找不到所需文件: {e}")
    exit()

# =========================================================================
# === [最終修正] 核心特徵代碼硬編碼 (基於實際注入的 Code) ===
# =========================================================================
# 這是修正的關鍵：直接使用 ADD.PY 注入到 aki_features.csv 中的精確 Code
CRITICAL_CODES_MAPPED = {
    'BUN': 'LAB//51006//mg/dL',
    'Creatinine': 'LAB//51106//mg/dL', # 注意：這裡的 Creatinine 可能是尿液肌酐 (Urine)
    'Lactate': 'LAB//50813//mmol/L', # 假設 Lactate 被成功注入
    'Potassium': 'LAB//51097//mEq/L'
}

# --- 3. 核心：特徵工程 (邏輯不變，但數據會成功創建列名) ---
print("\n--- 正在為每個病人構建【修正版】特徵矩陣 ---")

patient_ids = events_df['subject_id'].unique()
patient_features_df = pd.DataFrame(index=patient_ids)

# A. 處理數值型特徵 (mean/std)
numeric_events = events_df[events_df['code'].isin(selected_codes) & events_df['numeric_value'].notna()]
if not numeric_events.empty:
    stats = numeric_events.groupby(['subject_id', 'code'])['numeric_value'].agg(['mean', 'std']).unstack()
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    patient_features_df = patient_features_df.join(stats)
    print(f"已為 {len(stats.columns)} 個數值型特徵計算了統計量。")

# B/C. 處理其他特徵 (不變)
binary_events = events_df[events_df['code'].isin(selected_codes) & events_df['numeric_value'].isna()]
if not binary_events.empty:
    binary_flags = pd.pivot_table(binary_events, index='subject_id', columns='code', aggfunc='size', fill_value=0)
    binary_flags = (binary_flags > 0).astype(int)
    binary_flags.columns = [f"{col}_presence" for col in binary_flags.columns]
    patient_features_df = patient_features_df.join(binary_flags)
    print(f"已為 {len(binary_flags.columns)} 個特徵創建了存在性標誌。")
birth_df = events_df[events_df['code'] == 'MEDS_BIRTH'][['subject_id', 'time']].rename(columns={'time': 'birth_time'})
first_event_df = events_df.dropna(subset=['time']).groupby('subject_id')['time'].min().reset_index().rename(columns={'time': 'first_event_time'})
age_df = pd.merge(birth_df, first_event_df, on='subject_id').dropna()
patient_features_df['age'] = age_df.set_index('subject_id').apply(lambda row: (row['first_event_time'] - row['birth_time']).days / 365.25, axis=1).rename('age')
med_codes = inventory_df[inventory_df['code'].str.startswith(('INPUTEVENTS', 'PRESCRIPTIONS', 'PHARMACY'), na=False)]['code']
med_events = events_df[events_df['code'].isin(med_codes)]
patient_features_df['Medications_Count'] = med_events.groupby('subject_id')['code'].nunique()
diag_codes = inventory_df[inventory_df['code'].str.startswith('DIAGNOSIS', na=False)]['code']
diag_events = events_df[events_df['code'].isin(diag_codes)]
patient_features_df['Diagnoses_Count'] = diag_events.groupby('subject_id')['code'].nunique()
print("已添加年齡、藥物計數和診斷計數等聚合特徵。")


# --- 4. 數據預處理 (使用精確 Code 查找列名) ---
print("\n--- 正在進行【最終修正版】數據預處理 ---")
patient_features_df = patient_features_df.fillna(0)

# [優化1] 動態添加交互特徵 (BUN/Creatinine Ratio)
print("--- [優化1] 正在添加交互特徵 ---")
bun_code = CRITICAL_CODES_MAPPED['BUN']
creat_code = CRITICAL_CODES_MAPPED['Creatinine']

# [核心修正] 使用精確 Code 匹配聚合列名
bun_col_prefix = bun_code.replace('//', '_') 
creat_col_prefix = creat_code.replace('//', '_')

bun_mean_cols = [col for col in patient_features_df.columns if bun_col_prefix in col and 'mean' in col]
creat_mean_cols = [col for col in patient_features_df.columns if creat_col_prefix in col and 'mean' in col]

if bun_mean_cols and creat_mean_cols:
    bun_col = bun_mean_cols[0]
    creat_col = creat_mean_cols[0]
    patient_features_df['bun_creat_ratio'] = patient_features_df[bun_col] / (patient_features_df[creat_col] + 1e-6)
    print(f"已基於'{bun_col}'和'{creat_col}' 添加了 BUN/Creatinine Ratio 特徵。")
else:
    print(f"警告: 未能計算 BUN/Creatinine Ratio，因為在特徵矩陣中找不到 '{bun_code}' 或 '{creat_code}' 的均值列。")


# [優化2] 動態對傾斜數據進行對數變換
print("--- [優化2] 正在對傾斜數據進行對數變換 ---")
codes_to_transform = list(CRITICAL_CODES_MAPPED.values()) # 對所有關鍵數值 Code 進行變換

cols_to_transform = []
if codes_to_transform:
    all_aggregated_cols = [col for col in patient_features_df.columns if '_mean' in col or '_std' in col]
    for code in codes_to_transform:
        code_prefix = code.replace('//', '_') 
        # 查找所有包含 code_prefix 的 mean/std 聚合列
        cols_to_transform.extend([col for col in all_aggregated_cols if code_prefix in col])
    
    cols_to_transform = list(set(cols_to_transform))

if cols_to_transform:
    patient_features_df[cols_to_transform] = patient_features_df[cols_to_transform].clip(lower=0)
    for col in cols_to_transform:
        patient_features_df[col] = np.log1p(patient_features_df[col])
    print(f"已對 {len(cols_to_transform)} 個特徵進行對數變換。")
else:
    print("警告: 未找到指定用於對數變換的核心特徵列。")


# --- 4. 數據預處理 (標準化) ---
zero_variance_cols = patient_features_df.columns[patient_features_df.var() < 1e-6]
patient_features_df = patient_features_df.drop(columns=zero_variance_cols)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(patient_features_df)
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
print(f"預處理完成。最終特徵矩陣形狀: {features_tensor.shape}")


# --- 5, 6, 7. 訓練、聚類、保存 (保持不變) ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, embedding_dim))
        self.decoder = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, input_dim))
    def forward(self, x):
        encoded = self.encoder(x); decoded = self.decoder(encoded); return decoded
dataset = TensorDataset(features_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = Autoencoder(input_dim=features_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print("\n--- [優化3] 開始訓練【最終修正版】自編碼器模型 ---")
num_epochs = 150
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
for epoch in range(num_epochs):
    epoch_loss = 0
    for data in dataloader:
        inputs = data[0]; outputs = model(inputs); loss = criterion(outputs, inputs)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader); scheduler.step(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Reconstruction Loss: {avg_loss:.4f}')
print("訓練完成。")
print("\n--- 正在提取 Embedding 並進行 K-Means 聚類 ---")
model.eval()
with torch.no_grad():
    embedding_array = model.encoder(features_tensor).cpu().numpy()
print(f"成功生成 Embedding 數組！形狀為: {embedding_array.shape}")
print("--- [優化4] 正在使用肘部法則尋找最佳 K 值 ---")
inertia = []; k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto'); kmeans.fit(embedding_array); inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 5)); plt.plot(k_range, inertia, 'bo-'); plt.xlabel('Number of clusters (k)'); plt.ylabel('Inertia'); plt.title('Elbow Method For Optimal k'); plt.grid(True)
plt.savefig('elbow_method.png'); print("肘部圖已保存到 elbow_method.png。請查看此圖並確定最佳 K 值。"); plt.show()
try:
    optimal_k = int(input("根據肘部圖，請輸入您選擇的最佳 K 值: "))
except (ValueError, EOFError):
    print("輸入無效或在非交互模式下運行，將使用默認值 K=4"); optimal_k = 4
print(f"\n正在使用 K={optimal_k} 進行最終聚類...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(embedding_array)
print(f"\n K-Means 聚類完成，將病人分為 {optimal_k} 個簇。")
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index(); print("各簇的病人數量分佈:"); print(cluster_counts)
print("\n--- 正在保存 Embedding 數組和聚類標籤 ---")
patient_features_df['cluster_label'] = cluster_labels; patient_features_df.to_csv('patient_clusters_and_features.csv')
print("病人特徵及對應的聚類標籤已保存到: patient_clusters_and_features.csv")
np.save('patient_embedding_array.npy', embedding_array); np.save('patient_cluster_labels.npy', cluster_labels)
print("Embedding 數組已保存到: patient_embedding_array.npy"); print("聚類標籤數組已保存到: patient_cluster_labels.npy")