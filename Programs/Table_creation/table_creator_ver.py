import pandas as pd
import os

# === INPUT & OUTPUT PATHS ===
#input_csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Moving_vs_static\Verification_mov.csv"
#output_ods_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Graphs\Moving_vs_static\Verification_mov.ods"

input_csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Moving_vs_static\Verification_stat.csv"
output_ods_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Graphs\Moving_vs_static\Verification_stat.ods"

# Remove old output file if it exists
if os.path.exists(output_ods_path):
    os.remove(output_ods_path)

# === LOAD CSV ===
df = pd.read_csv(input_csv_path)

# === EXTRACT SPLIT TYPE ===
df['Split'] = df['Model'].str.extract(r'\((.*?)\)')

# === SELECT ONLY RELEVANT COLUMNS ===
cols_to_keep = ['Model', 'Test Accuracy', 'Precision', 'Recall', 'Specificity', 'EER']

# === SPLIT DATAFRAMES ===
table_80_20 = df[df['Split'] == '80/20'][cols_to_keep]
table_s1s2_s3 = df[df['Split'] == 'S1+S2 vs S3'][cols_to_keep]

# === MARGIN SETTINGS ===
top_margin = 3
left_margin = 3
spacing_between_tables = 2  # empty rows between tables

# === SAVE BOTH TABLES IN SINGLE SHEET ===
with pd.ExcelWriter(output_ods_path, engine="odf") as writer:
    # 80/20 split table
    table_80_20.to_excel(
        writer,
        sheet_name="Results",
        startrow=top_margin,
        startcol=left_margin,
        index=False
    )
    
    # S1+S2 vs S3 split table, below the first
    table_s1s2_s3.to_excel(
        writer,
        sheet_name="Results",
        startrow=top_margin + len(table_80_20) + spacing_between_tables,
        startcol=left_margin,
        index=False
    )

print(f"ODS file saved at: {output_ods_path}")
