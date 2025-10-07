import os
import pandas as pd

# === CONFIGURATION ===
input_file = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Machine_Learning_results\Identification_single_results\selected_features_ft.csv"

# === Read the single input file ===
df = pd.read_csv(input_file)

# Ensure correct columns exist
if not {'Model', 'Feature', 'F-score'}.issubset(df.columns):
    raise ValueError("The input file must contain 'Model', 'Feature', and 'F-score' columns.")

# Exclude Naive Bayes models
df = df[~df['Model'].str.contains('Naive Bayes', case=False, na=False)]

# === Compute cumulative F-scores ===
feature_scores = df.groupby('Feature')['F-score'].sum().reset_index()
feature_scores = feature_scores.sort_values(by='F-score', ascending=False).reset_index(drop=True)
feature_scores.rename(columns={'F-score': 'Cumulative F-score'}, inplace=True)

# === Save cumulative results ===
output_path = os.path.join(os.path.dirname(input_file), "cumulative_fscore_results.csv")
feature_scores.to_csv(output_path, index=False)

# === Identify missing features ===
all_features = [f"f{i}" for i in range(83)]
present_features = set(feature_scores['Feature'])
missing_features = [f for f in all_features if f not in present_features]

missing_df = pd.DataFrame(missing_features, columns=["Missing Features"])
missing_output_path = os.path.join(os.path.dirname(input_file), "missing_features.csv")
missing_df.to_csv(missing_output_path, index=False)

# === Print summary ===
print(f"✅ Cumulative F-score results saved to:\n{output_path}")
print(f"✅ Missing features list saved to:\n{missing_output_path}\n")
print("Top 10 features:\n")
print(feature_scores.head(10).to_string(index=False))
