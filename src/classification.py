import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_DIR = "data/TCGA-PANCAN-HiSeq-801x20531"

data = pd.read_csv(f"{DATA_DIR}/data.csv")
labels = pd.read_csv(f"{DATA_DIR}/labels.csv")

# Remove sample ID
X = data.drop(columns=["Unnamed: 0"])

# BRCA = 1
# Other cancer types = 0
y = (labels["Class"] == "BRCA").astype(int)

print("=" * 60)
print("NAIVE BAYES CLASSIFICATION")
print("=" * 60)

print("\nOriginal data shape:")
print(X.shape)

print("\nTarget distribution:")
print(y.value_counts())


# --------------------------------------------------
# 2. Remove constant genes
# --------------------------------------------------

constant_remover = VarianceThreshold(threshold=0.0)

X = constant_remover.fit_transform(X)

print("\nData shape after removing constant genes:")
print(X.shape)


# --------------------------------------------------
# 3. Train / Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# --------------------------------------------------
# 4. Build leakage-safe pipeline
# --------------------------------------------------

model = Pipeline([
    (
        "variance_filter",
        VarianceThreshold(threshold=0.0)
    ),
    (
        "feature_selection",
        SelectKBest(
            score_func=f_classif,
            k=50
        )
    ),
    (
        "classifier",
        GaussianNB()
    )
])


# --------------------------------------------------
# 5. 5-Fold Stratified Cross-Validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 60)

print(f"\nAccuracy : {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f}")
print(f"Recall   : {cv_results['test_recall'].mean():.4f}")
print(f"F1-score : {cv_results['test_f1'].mean():.4f}")
print(f"ROC-AUC  : {cv_results['test_roc_auc'].mean():.4f}")


# --------------------------------------------------
# 6. Train final model
# --------------------------------------------------

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 7. Test set evaluation
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_probability):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Other Cancer", "BRCA"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))