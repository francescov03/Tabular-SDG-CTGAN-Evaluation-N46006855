import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

file_path = "../dataset/bank-additional-full.csv"
output_path = "../dataset/bank_synthetic.csv"

df = pd.read_csv(file_path, sep=";")

print("Shape dataset reale:", df.shape)
print("\nPrime righe:")
print(df.head())

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

synthesizer = CTGANSynthesizer(
    metadata=metadata,
    epochs=50
)

print("\nInizio training CTGAN...")
synthesizer.fit(df)

print("\nTraining completato.")

synthetic_df = synthesizer.sample(num_rows=len(df))

print("\nShape dataset sintetico:", synthetic_df.shape)
print("\nPrime righe sintetiche:")
print(synthetic_df.head())

synthetic_df.to_csv(output_path, index=False)
print(f"\nDataset sintetico salvato in: {output_path}")