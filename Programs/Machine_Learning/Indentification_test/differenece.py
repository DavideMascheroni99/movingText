import pandas as pd

# Path al file
PATH = r"C:\Users\david\OneDrive\Documenti\Tesi_BehavBio\Programs\Programs\Machine_Learning\Indentification_test\Identification_single_different.csv" 

# Carica
df = pd.read_csv(PATH)

# Estrai Classifier e Split dal nome modello
df["Classifier"] = df["Model"].str.extract(r"^(.*?) \(")
df["Split"] = df["Model"].str.extract(r"\((.*?)\)$")

# Teniamo solo le colonne rilevanti
df = df[["Animation", "Classifier", "Split", "Test Accuracy"]]

# Portiamo in formato wide: una colonna per 80/20 e una per S1+S2 vs S3
wide = (
    df.pivot_table(index=["Animation", "Classifier"],
                   columns="Split",
                   values="Test Accuracy",
                   aggfunc="first")
    .rename_axis(None, axis=1)
)

# Verifica che entrambe le split esistano
required_cols = {"80/20", "S1+S2 vs S3"}
if not required_cols.issubset(wide.columns):
    missing = required_cols - set(wide.columns)
    raise ValueError(f"Mancano queste split nel file: {missing}")

# Differenza di accuratezza
wide["Δ(80/20 − S1+S2vsS3)"] = wide["80/20"] - wide["S1+S2 vs S3"]

# Ordina e arrotonda
out = (
    wide.reset_index()
        .sort_values(["Animation", "Classifier"])
)
out["Δ(80/20 − S1+S2vsS3)"] = out["Δ(80/20 − S1+S2vsS3)"].round(4)

# Mostra a video solo ciò che serve
print(out[["Animation", "Classifier", "Δ(80/20 − S1+S2vsS3)"]])
