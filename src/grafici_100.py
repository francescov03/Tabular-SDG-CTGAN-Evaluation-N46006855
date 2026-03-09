import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("../output", exist_ok=True)

real_df = pd.read_csv("../dataset/bank-additional-full.csv", sep=";")
syn_df = pd.read_csv("../dataset/bank_synthetic_100epoche.csv")

sns.set(style="whitegrid")

# ----------------------------
# 1. Istogrammi variabili numeriche
# ----------------------------
numerical_cols = ["age", "duration", "campaign", "euribor3m"]

for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    plt.hist(real_df[col], bins=30, alpha=0.6, label="Reale", density=True)
    plt.hist(syn_df[col], bins=30, alpha=0.6, label="Sintetico", density=True)
    plt.title(f"Confronto distribuzione - {col}")
    plt.xlabel(col)
    plt.ylabel("Densità")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../output/hist_{col}_100epoche.png")
    plt.close()

# ----------------------------
# 2. Bar plot variabili categoriche
# ----------------------------
categorical_cols = ["job", "marital", "education", "y"]

for col in categorical_cols:
    real_freq = real_df[col].value_counts(normalize=True).sort_index()
    syn_freq = syn_df[col].value_counts(normalize=True).sort_index()

    freq_df = pd.DataFrame({
        "Reale": real_freq,
        "Sintetico": syn_freq
    }).fillna(0)

    freq_df.plot(kind="bar", figsize=(10, 5))
    plt.title(f"Confronto frequenze - {col}")
    plt.xlabel(col)
    plt.ylabel("Proporzione")
    plt.tight_layout()
    plt.savefig(f"../output/bar_{col}_100epoche.png")
    plt.close()

# ----------------------------
# 3. Heatmap correlazioni numeriche
# ----------------------------
real_num = real_df.select_dtypes(include=["int64", "float64"])
syn_num = syn_df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 8))
sns.heatmap(real_num.corr(), cmap="coolwarm", center=0)
plt.title("Heatmap correlazioni - Dataset reale")
plt.tight_layout()
plt.savefig("../output/heatmap_reale_100epoche.png")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(syn_num.corr(), cmap="coolwarm", center=0)
plt.title("Heatmap correlazioni - Dataset sintetico")
plt.tight_layout()
plt.savefig("../output/heatmap_sintetico_100epoche.png")
plt.close()

print("Grafici salvati nella cartella ../output")
