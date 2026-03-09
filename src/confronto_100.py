import pandas as pd

real_df = pd.read_csv("../dataset/bank-additional-full.csv", sep=";")
syn_df = pd.read_csv("../dataset/bank_synthetic_100epoche.csv")

print("Shape reale:", real_df.shape)
print("Shape sintetico:", syn_df.shape)

print("\nColonne uguali:", list(real_df.columns) == list(syn_df.columns))

print("\n--- Confronto statistiche numeriche ---")
num_cols = real_df.select_dtypes(include=["int64", "float64"]).columns

for col in num_cols:
    print(f"\nColonna: {col}")
    print("Reale:")
    print(real_df[col].describe())
    print("Sintetico:")
    print(syn_df[col].describe())

print("\n--- Confronto frequenze categoriche ---")
cat_cols = real_df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    print(f"\nColonna: {col}")
    print("Reale:")
    print(real_df[col].value_counts(normalize=True).head(10))
    print("Sintetico:")
    print(syn_df[col].value_counts(normalize=True).head(10))
