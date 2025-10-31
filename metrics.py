import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score as performance_measure
from sklearn.metrics import roc_auc_score
import warnings
from typing import List, Tuple

warnings.filterwarnings('ignore')

def evaluate_model_performance(
    result_select: pd.DataFrame,
    target_model: str,
    models: List[str],
    round_num: int = 200
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates AUPRC and AUROC performance metrics, estimates variance using 
    bootstrap resampling, and performs a paired t-test against the target model.
    
    Args:
        result_select (pd.DataFrame): DataFrame containing 'Label' and model prediction scores for a subgroup.
        target_model (str): The name of the model column to be used as the benchmark.
        models (List[str]): List of all model score columns to evaluate.
        round_num (int): Number of bootstrap resampling iterations.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - Performance_General: Final metrics (AUPRC, AUROC, CI) for all models.
            - Two_Size_P_Record: Two-sided p-values comparing each model against the target model (AUPRC metric).
    """
    
    # --- 1. Initialization and Final Performance Calculation ---
    
    performance_result_auprc = pd.DataFrame(index=range(round_num), columns=models)
    performance_general = pd.DataFrame(
        index=models,
        columns=['final_AUPRC', 'final_AUROC', 'mean', 'std', '95%_upper', '95%_lower']
    )
    
    labels = result_select['Label']
    
    for model_name in models:
        scores = result_select.loc[:, model_name]
        
        # Calculate AUPRC (Aliased as performance_measure in original code)
        final_auprc = performance_measure(labels, scores)
        # Calculate AUROC (New addition)
        final_auroc = roc_auc_score(labels, scores)
        
        performance_general.loc[model_name, 'final_AUPRC'] = final_auprc
        performance_general.loc[model_name, 'final_AUROC'] = final_auroc # AUROC stored here

    # --- 2. Bootstrap Resampling ---
    
    for round_id in range(round_num):
        # Sample with replacement (Bootstrap)
        sample_result = result_select.sample(frac=1, replace=True)
        sample_labels = sample_result['Label']
        
        for model_name in models:
            sample_scores = sample_result.loc[:, model_name]
            # Calculate AUPRC for the sample
            performance_auprc = performance_measure(sample_labels, sample_scores)
            performance_result_auprc.loc[round_id, model_name] = performance_auprc

    # --- 3. Calculate Mean, Std, and Confidence Intervals (Based on AUPRC) ---
    
    # Rename 'final_performance' column to 'final_AUPRC' for clarity and consistency
    performance_general = performance_general.rename(columns={'final_performance': 'final_AUPRC'}) 

    performance_general.loc[:, 'mean'] = performance_result_auprc.mean(axis=0)
    performance_general.loc[:, 'std'] = performance_result_auprc.std(axis=0)
    
    # Calculate 95% Confidence Intervals
    performance_general['95%_upper'] = performance_general['final_AUPRC'] + (performance_general['std'] * 1.96)
    performance_general['95%_lower'] = performance_general['final_AUPRC'] - (performance_general['std'] * 1.96)
    
    # --- 4. Paired t-test (Comparison against target_model, using AUPRC) ---
    
    p_score_record = pd.DataFrame(index=models, columns=['t_score'])
    
    for model_name in models:
        
        # Calculate covariance between the current model and the target model
        cov_between_models = (
            np.mean(performance_result_auprc.loc[:, model_name].values * performance_result_auprc.loc[:, target_model].values) - 
            (performance_general.loc[model_name, 'mean'] * performance_general.loc[target_model, 'mean'])
        )
        
        # Calculate the t-statistic (P-score in original code)
        numerator = (
            performance_general.loc[target_model, 'final_AUPRC'] - 
            performance_general.loc[model_name, 'final_AUPRC']
        )
        
        # Denominator: Std Dev of the difference
        denominator = np.sqrt(
            (performance_general.loc[target_model, 'std'] ** 2) + 
            (performance_general.loc[model_name, 'std'] ** 2) - 
            (2 * cov_between_models)
        )
        
        if denominator != 0:
            t_score = numerator / denominator
            p_score_record.loc[model_name, 't_score'] = t_score
        else:
            p_score_record.loc[model_name, 't_score'] = 0

    # 5. Convert t-score to two-sided p-value
    one_size_p_record = norm.sf(abs(p_score_record['t_score'].fillna(0)))
    two_size_p_record = 2 * one_size_p_record
    
    # Format final P-value DataFrame
    two_size_p_record = pd.DataFrame(two_size_p_record, index=models, columns=['p_value'])

    return performance_general, two_size_p_record


# --- Example Usage (Mirroring your original loop structure) ---
if __name__ == '__main__':
    
    # MOCK DATA SETUP: Replace with your actual file loading in a real environment
    N = 1000
    np.random.seed(42)
    
    result = pd.DataFrame({
        'Drg': np.repeat([100, 200, 300], N // 3),
        'Label': np.random.randint(0, 2, N),
        'update_1921_mat_proba': np.random.uniform(0.1, 0.9, N) + 0.1 * np.random.rand(N),
        'model_A': np.random.uniform(0.1, 0.9, N) + 0.05 * np.random.rand(N),
        'model_B': np.random.uniform(0.1, 0.9, N) + 0.12 * np.random.rand(N),
    })
    
    disease_list = pd.DataFrame({0: [100, 200, 300]}) # Mock DRG codes
    
    target_model = 'update_1921_mat_proba'
    models = result.iloc[:, 2:].columns.tolist() # Scores start at column 2

    # Global storage DataFrames mirroring your original code
    disease_score_record = pd.DataFrame()
    disease_performance_record = pd.DataFrame()

    print("Starting Subgroup Evaluation...")
    print("-" * 30)

    for disease_num in range(disease_list.shape[0]):
        
        drg_code = disease_list.iloc[disease_num, 0]
        
        # Select the subgroup data (Matching your original script logic)
        disease_true = result.loc[:,'Drg'] == drg_code
        result_select = result.loc[disease_true].copy()
        
        if result_select.empty:
            print(f"Skipping DRG {drg_code}: No data found.")
            continue
            
        print(f"Processing Subgroup DRG{drg_code} (N={len(result_select)})...")
        
        # CALL THE ADAPTED FUNCTION
        performance_general, p_values_df = evaluate_model_performance(
            result_select=result_select,
            target_model=target_model,
            models=models,
            round_num=100
        )
        
        # Store final AUPRC results (Matching your original script logic)
        for model_name in models:
             disease_performance_record.loc[f'Drg{drg_code}', model_name] = performance_general.loc[model_name, 'final_AUPRC']

        # Store P-score (t-score) results (Matching your original script logic)
        for model_name in models:
            if model_name != target_model:
                disease_score_record.loc[f'Drg{drg_code}', model_name] = p_values_df.loc[model_name, 'p_value']
            else:
                 disease_score_record.loc[f'Drg{drg_code}', model_name] = np.nan # Target model vs itself has no p-value

    # Final P-value calculation (Matching your original script logic)
    # NOTE: The p-value stored in disease_score_record is already the two-sided p-value from the function output.
    # The original code's final block is redundant if the p-value is calculated inside the loop, 
    # but we will mimic the final storage logic structure.
    
    # We rename it to the final output file to avoid the redundant final calculation
    two_size_p_record = disease_score_record 

    print("\nEvaluation Summary:")
    print("-" * 30)
    print("Final AUPRC Records (disease_performance_record):")
    print(disease_performance_record)
    print("\nFinal P-value Records (two_size_p_record):")
    print(two_size_p_record)

    # Mimic final file saving
    # two_size_p_record.to_csv("/blue/mei.liu/yujingju/aki-project/AUPRC_compare_top31_p.csv")
    # disease_performance_record.to_csv("/blue/mei.liu/yujingju/aki-project/AUPRC_compare_top31_ori.csv")