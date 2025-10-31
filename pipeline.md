# Transform from ETL to MEDS
mkdir ./my_meds_demo_output
MEDS_extract-MIMIC_IV root_output_dir=./my_meds_demo_output do_demo=True do_copy=True

# Clustering
trash.py
python check.py
python select_features.py
python verify.py
python add.py
python clustering.py
python similar_patients_rag.py