import pandas as pd

# --- User-defined paths ---
input_csv_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Prova_folder\Verification_single_intruders_results_two_st.csv"
output_ods_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Graphs\Verification_single_intruders_two_st.ods"

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
    for animation in df['Animation'].unique():
        # Filter for animation and remove Naive Bayes
        sub_df = df[(df['Animation'] == animation) & (df['Model'] != 'Naive Bayes')][['Model'] + metrics].copy()
        sub_df = sub_df.rename(columns={'Test Accuracy': 'Accuracy'})
        
        # Convert numeric columns to float and round
        for col in ['Accuracy', 'Recall', 'Specificity', 'Precision', 'EER']:
            sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce').round(3)
        
        # Step 1: Add 2 empty columns on the left
        sub_df.insert(0, " ", "")  # first empty column
        sub_df.insert(0, "  ", "")  # second empty column
        
        # Step 2: Add header row as first row
        header_row = pd.DataFrame([sub_df.columns.tolist()], columns=sub_df.columns)
        sub_df = pd.concat([header_row, sub_df], ignore_index=True)
        
        # Step 3: Add 3 empty rows at the very top
        empty_rows = pd.DataFrame([[""]*len(sub_df.columns)]*3, columns=sub_df.columns)
        final_df = pd.concat([empty_rows, sub_df], ignore_index=True)
        
        # Step 4: Write to ODS sheet without adding another header
        final_df.to_excel(writer, sheet_name=animation, index=False, header=False)

print(f"ODS file saved as '{output_ods_path}'")
