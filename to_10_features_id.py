import os
import pandas as pd

# === CONFIGURATION ===
folder_path = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Identification_results\Identification_KBest\Feature"

# Initialize cumulative dictionary
feature_scores = {}

# Read all CSV files
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath)

        # Ensure expected columns exist
        if {'Feature', 'F-score'}.issubset(df.columns):
            for _, row in df.iterrows():
                feature = row['Feature']
                score = row['F-score']
                feature_scores[feature] = feature_scores.get(feature, 0) + score

# === Compute cumulative results ===
result_df = pd.DataFrame(list(feature_scores.items()), columns=["Feature", "Cumulative F-score"])
result_df = result_df.sort_values(by="Cumulative F-score", ascending=False).reset_index(drop=True)

# === Save cumulative results ===
output_path = os.path.join(folder_path, "cumulative_fscore_results.csv")
result_df.to_csv(output_path, index=False)

# === Identify missing features ===
# Assuming features go from f0 to f82 (adjust if necessary)
all_features = [f"f{i}" for i in range(83)]
present_features = set(feature_scores.keys())
missing_features = [f for f in all_features if f not in present_features]

missing_df = pd.DataFrame(missing_features, columns=["Missing Features"])
missing_output_path = os.path.join(folder_path, "missing_features.csv")
missing_df.to_csv(missing_output_path, index=False)

# === Print summary ===
print(f"✅ Cumulative F-score results saved to:\n{output_path}")
print(f"✅ Missing features list saved to:\n{missing_output_path}\n")
print("Top 10 features:\n")
print(result_df.head(10).to_string(index=False))
