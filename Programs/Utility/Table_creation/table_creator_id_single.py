import pandas as pd
import os

# === INPUT & OUTPUT PATHS ===
#input_csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\Identification_single_results_st.csv"
#output_ods_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Graphs\Identification_single\Identification_single_st.ods"

input_csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\Identification_single_results_ft.csv"
output_ods_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Graphs\Identification_single\Identification_single_ft.ods"

if os.path.exists(output_ods_path):
    os.remove(output_ods_path)

# === LOAD CSV ===
df = pd.read_csv(input_csv_path)

# === FILTER OUT NAIVE BAYES ===
df = df[~df['Model'].str.startswith('Naive Bayes')]

# Extract classifier names and split type
df['Classifier'] = df['Model'].str.extract(r'^(.*?)\s*\(')
df['Split'] = df['Model'].str.extract(r'\((.*?)\)')

# Create pivot tables with flipped axes (classifiers on rows, animations on columns)
pivot_80_20 = df[df['Split'] == '80/20'].pivot(index='Classifier', columns='Animation', values='Test Accuracy')
pivot_s1s2_s3 = df[df['Split'] == 'S1+S2 vs S3'].pivot(index='Classifier', columns='Animation', values='Test Accuracy')

# === MARGIN SETTINGS ===
top_margin = 5
left_margin = 4  

# Save to ODS in the same sheet with margins
with pd.ExcelWriter(output_ods_path, engine="odf") as writer:
    pivot_80_20.to_excel(writer, sheet_name="Results", startrow=top_margin, startcol=left_margin)
    pivot_s1s2_s3.to_excel(writer, sheet_name="Results", startrow=top_margin + len(pivot_80_20) + 4, startcol=left_margin)

print(f"ODS file saved at: {output_ods_path}")
