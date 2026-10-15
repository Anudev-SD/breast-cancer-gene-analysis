import pandas as pd

# Dataset location
DATA_DIR = "data/TCGA-PANCAN-HiSeq-801x20531"

# Load data
data = pd.read_csv(f"{DATA_DIR}/data.csv")
labels = pd.read_csv(f"{DATA_DIR}/labels.csv")

# Remove sample ID columns
X = data.drop(columns=["Unnamed: 0"])
y = labels["Class"]

print("=" * 60)
print("PREPROCESSING AND DATASET CHECK")
print("=" * 60)

print("\nGene expression data shape:")
print(X.shape)

print("\nTarget labels shape:")
print(y.shape)

print("\nCancer classes:")
print(y.value_counts())

print("\nMissing values in gene expression data:")
print(X.isnull().sum().sum())

print("\nMissing values in labels:")
print(y.isnull().sum())

print("\nFirst 5 target labels:")
print(y.head())