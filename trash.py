import pandas as pd

MEDS_COHORT_DIR = './my_meds_demo_output/MEDS_cohort'
DATA_PATH = f'{MEDS_COHORT_DIR}/data/train/0.parquet'
CODES_PATH = f'{MEDS_COHORT_DIR}/metadata/codes.parquet'

events_df = pd.read_parquet(DATA_PATH)
codes_df = pd.read_parquet(CODES_PATH)
unique_codes_in_data = pd.DataFrame({'code': events_df['code'].unique()})
feature_inventory = pd.merge(unique_codes_in_data, codes_df[['code', 'description']], on='code', how='left')
feature_inventory.to_csv('feature_inventory.csv', index=False)
print("特徵清單已保存到 feature_inventory.csv")