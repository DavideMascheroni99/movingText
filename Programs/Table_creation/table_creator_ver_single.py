import pandas as pd

# --- User-defined paths ---
input_csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Verification_single_results\Verification_single_results_ft.csv"
output_ods_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Graphs\Verification_single\Verification_single_ft.ods"

# Step 1: Read CSV
df = pd.read_csv(input_csv_path)

# Step 1b: Strip column names to avoid extra spaces
df.columns = df.columns.str.strip()

# Step 2: Define the metrics we want
metrics = ['Test Accuracy', 'Recall', 'Specificity', 'Precision', 'EER']

# Step 2b: Ensure metrics exist in the CSV
missing_cols = [col for col in metrics if col not in df.columns]
if missing_cols:
    raise ValueError(f"The following columns are missing in the CSV: {missing_cols}")

# Step 3: Create an ODS writer using the 'odf' engine
with pd.ExcelWriter(output_ods_path, engine='odf') as writer:
    # Step 4: Iterate over each unique animation
    for animation in df['Animation'].unique():
        sub_df = df[df['Animation'] == animation][['Model'] + metrics].copy()
        sub_df = sub_df.rename(columns={'Test Accuracy': 'Accuracy'})
        # Convert numeric columns to float and round to 3 decimals
        for col in ['Accuracy', 'Recall', 'Specificity', 'Precision', 'EER']:
            sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce').round(3)
        # Write to ODS sheet
        sub_df.to_excel(writer, sheet_name=animation, index=False)

print(f"ODS file saved as '{output_ods_path}'")
