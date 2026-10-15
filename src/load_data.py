import pandas as pd

# Dataset location
DATA_DIR = "data/TCGA-PANCAN-HiSeq-801x20531"

# Load the two CSV files
data = pd.read_csv(f"{DATA_DIR}/data.csv")
labels = pd.read_csv(f"{DATA_DIR}/labels.csv")

# Display basic information
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nGene expression data shape:")
print(data.shape)

print("\nLabels data shape:")
print(labels.shape)

print("\nFirst 5 rows of gene expression data:")
print(data.head())

print("\nFirst 5 rows of labels:")
print(labels.head())

print("\nGene expression data columns:")
print(data.columns[:10].tolist())

print("\nLabel columns:")
print(labels.columns.tolist())