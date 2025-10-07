import pandas as pd
import os

# === INPUT & OUTPUT PATHS ===
#input_csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Moving_vs_static\Identification_mov.csv"
#output_ods_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Graphs\Moving_vs_static\Identification_mov.ods"

input_csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Moving_vs_static\Identification_stat.csv"
output_ods_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Graphs\Moving_vs_static\Identification_stat.ods"

# Remove old output file if it exists
if os.path.exists(output_ods_path):
    os.remove(output_ods_path)

# === LOAD CSV ===
df = pd.read_csv(input_csv_path)

# === EXTRACT CLASSIFIER NAMES AND SPLIT TYPE ===
df['Classifier'] = df['Model'].str.extract(r'^(.*?)\s*\(')
df['Split'] = df['Model'].str.extract(r'\((.*?)\)')

# === CREATE TWO TABLES WITH REQUIRED COLUMNS ===
table_80_20 = df[df['Split'] == '80/20'][['Classifier', 'Split', 'Test Accuracy']]
table_s1s2_s3 = df[df['Split'] == 'S1+S2 vs S3'][['Classifier', 'Split', 'Test Accuracy']]

# === MARGIN SETTINGS ===
top_margin = 3
left_margin = 3

# === SAVE BOTH TABLES IN SINGLE SHEET ===
with pd.ExcelWriter(output_ods_path, engine="odf") as writer:
    table_80_20.to_excel(writer, sheet_name="Results", startrow=top_margin, startcol=left_margin, index=False)
    table_s1s2_s3.to_excel(
        writer,
        sheet_name="Results",
        startrow=top_margin + len(table_80_20) + 2,  # leave 2 empty rows between tables
        startcol=left_margin,
        index=False
    )

print(f"ODS file saved at: {output_ods_path}")
