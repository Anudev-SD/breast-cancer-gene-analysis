# 🧬 Breast Cancer Gene Expression Analysis

Classifying **breast cancer (BRCA)** from gene expression data using **Naive Bayes** and exploring gene relationships with a **Bayesian Network**.

The project uses the TCGA Pan-Cancer RNA-Seq dataset (801 samples × 20,531 genes) and treats the task as a binary problem: **BRCA vs. other cancer types**.

> ⚠️ **Disclaimer:** This is a learning and portfolio project. It is not a diagnostic tool and must not be used for medical decisions.

## ✨ What this project does

- **Feature selection**: picks the most informative genes with an ANOVA F-test (`SelectKBest`)
- **Classification**: trains a Gaussian Naive Bayes model inside a leakage-safe scikit-learn pipeline
- **Evaluation**: 5-fold stratified cross-validation plus a held-out test set (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
- **Bayesian Network**: discretizes the top genes, learns the network structure with the PC algorithm, estimates conditional probabilities, and runs probabilistic inference for BRCA

## 📊 Dataset

**TCGA-PANCAN-HiSeq-801x20531** (UCI Machine Learning Repository: *Gene Expression Cancer RNA-Seq*)

- 801 tumor samples, 20,531 gene expression features
- 5 cancer types, here grouped as **BRCA = 1** and **all other types = 0**

The data is **not included in this repo** (the main file is over 200 MB). To run the project:

1. Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna-seq)
2. Place the files like this:

```
data/
└── TCGA-PANCAN-HiSeq-801x20531/
    ├── data.csv
    └── labels.csv
```

## 📁 Project structure

```
Breast_Cancer_Gene_Analysis/
├── src/
│   ├── inspect_data.py        # Lists dataset files
│   ├── load_data.py           # Loads data and prints basic info
│   ├── preprocessing.py       # Shapes, class counts, missing-value check
│   ├── feature_selection.py   # Train/test split + top 50 genes (F-test)
│   ├── classification.py      # Naive Bayes pipeline, CV and test metrics
│   └── bayesian_network.py    # PC structure learning, CPDs, inference
├── results/
│   ├── bayesian_network.png
│   ├── bayesian_network_edges.csv
│   ├── bayesian_network_selected_genes.csv
│   ├── bayesian_network_cpds.txt
│   └── bayesian_inference.txt
├── requirements.txt
└── README.md
```

## 🚀 Getting started

### 1. Clone the repo

```bash
git clone https://github.com/Anudev-SD/breast-cancer-gene-analysis.git
cd breast-cancer-gene-analysis
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Follow the steps in the [Dataset](#-dataset) section.

### 5. Run the scripts

Run everything from the **project root** (the folder that contains `src/` and `data/`):

```bash
python src/preprocessing.py
python src/feature_selection.py
python src/classification.py
python src/bayesian_network.py
```

The Bayesian network script writes its outputs to `results/`.

## 🔬 Methods

### Naive Bayes classification (`classification.py`)

1. Convert labels to binary (BRCA = 1, others = 0)
2. Split data 80/20 with stratification (`random_state=42`)
3. Build a pipeline: **variance filter → SelectKBest (k = 50) → GaussianNB**
4. Evaluate with 5-fold stratified cross-validation on the training set
5. Fit on the full training set and report metrics on the held-out test set

Feature selection lives **inside** the pipeline, so it is re-fit within each fold. This avoids data leakage into the validation data.

### Bayesian Network (`bayesian_network.py`)

1. Split the data and remove constant genes using the training set only
2. Select the top 5 genes with the F-test
3. Discretize each gene into 3 levels (low / medium / high) using quantile bins
4. Learn the structure with the **PC algorithm** (stable variant, chi-square test, α = 0.05)
5. Estimate conditional probability tables with a **BDeu** prior
6. Run inference with **Variable Elimination**

**Example result** (from `results/bayesian_inference.txt`):

| Query | P(BRCA = 1) |
|---|---|
| Prior (no evidence) | 0.428 |
| Posterior given high expression of `gene_6876` and `gene_7964` | 0.740 |

Learned network (see `results/bayesian_network.png`): `gene_6876` and `gene_18746` have edges into `BRCA`, while `BRCA` connects onward to `gene_17801`.

> The edge directions come from the PC algorithm and show statistical dependence, not proven causation.

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · pgmpy · NetworkX · matplotlib · seaborn

## 🔮 Future improvements

- Compare Naive Bayes with other models (logistic regression, random forest)
- Multi-class classification across all 5 cancer types
- Try different numbers of selected genes and discretization levels
- Add a notebook with visualizations of the results

## 📄 License

MIT. Feel free to use this project for learning.
