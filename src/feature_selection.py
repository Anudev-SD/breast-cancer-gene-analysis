import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_DIR = "data/TCGA-PANCAN-HiSeq-801x20531"

data = pd.read_csv(f"{DATA_DIR}/data.csv")
labels = pd.read_csv(f"{DATA_DIR}/labels.csv")

# Remove sample ID
X = data.drop(columns=["Unnamed: 0"])

# BRCA = 1, all other cancer types = 0
y = (labels["Class"] == "BRCA").astype(int)

print("=" * 60)
print("FEATURE SELECTION")
print("=" * 60)

print("\nOriginal feature shape:")
print(X.shape)

print("\nTarget distribution:")
print(y.value_counts())


# --------------------------------------------------
# 2. Split data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:")
print(X_train.shape)

print("\nTesting set:")
print(X_test.shape)


# --------------------------------------------------
# 3. Select the most informative genes
# --------------------------------------------------

K = 50

selector = SelectKBest(
    score_func=f_classif,
    k=K
)

X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected gene names
selected_genes = X_train.columns[selector.get_support()]

print("\nNumber of selected genes:")
print(len(selected_genes))

print("\nSelected genes:")
print(selected_genes.tolist())

print("\nSelected training data shape:")
print(X_train_selected.shape)

print("\nSelected testing data shape:")
print(X_test_selected.shape)