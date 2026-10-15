import os
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from pgmpy.estimators import PC, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/TCGA-PANCAN-HiSeq-801x20531"
RESULTS_DIR = "results"
N_GENES = 5
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

data_path = os.path.join(DATA_DIR, "data.csv")
labels_path = os.path.join(DATA_DIR, "labels.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Could not find {data_path}. Run this script from the project root."
    )
if not os.path.exists(labels_path):
    raise FileNotFoundError(
        f"Could not find {labels_path}. Run this script from the project root."
    )

data = pd.read_csv(data_path)
labels = pd.read_csv(labels_path)

if "Class" not in labels.columns:
    raise ValueError("labels.csv must contain a 'Class' column.")

# Drop the sample-ID column from expression data, if present.
X = data.drop(columns=["Unnamed: 0"], errors="ignore")

# BRCA = 1; all other cancer types = 0.
y = (labels["Class"] == "BRCA").astype(int)

if len(X) != len(y):
    raise ValueError(
        f"Expression rows ({len(X)}) and label rows ({len(y)}) do not match."
    )

# Ensure expression features are numeric.
X = X.apply(pd.to_numeric, errors="coerce")

# Remove columns that contain no usable values, then fill any remaining
# missing values using each column's median.
X = X.dropna(axis=1, how="all")
X = X.fillna(X.median()).fillna(0)

print("=" * 60)
print("BREAST CANCER BAYESIAN NETWORK ANALYSIS")
print("=" * 60)
print("\nOriginal gene-expression data:")
print(X.shape)
print("\nTarget distribution:")
print(y.value_counts().sort_index())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining data:")
print(X_train.shape)
print("\nTesting data:")
print(X_test.shape)


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

variance_filter = VarianceThreshold(threshold=0.0)
X_train_filtered_array = variance_filter.fit_transform(X_train)

training_genes = X_train.columns[variance_filter.get_support()]
X_train_filtered = pd.DataFrame(
    X_train_filtered_array,
    columns=training_genes,
    index=X_train.index,
)

print("\nTraining data after removing constant genes:")
print(X_train_filtered.shape)

if X_train_filtered.shape[1] < 1:
    raise ValueError("No non-constant genes remain after variance filtering.")


# ============================================================
# FEATURE SELECTION
# ============================================================

k = min(N_GENES, X_train_filtered.shape[1])
selector = SelectKBest(score_func=f_classif, k=k)
X_train_selected_array = selector.fit_transform(X_train_filtered, y_train)

selected_genes = training_genes[selector.get_support()]
selected_data = pd.DataFrame(
    X_train_selected_array,
    columns=selected_genes,
    index=X_train.index,
)

print("\nSelected genes:")
for gene in selected_genes:
    print("-", gene)


# ============================================================
# ADD TARGET VARIABLE
# ============================================================

# Use the labels aligned to the training rows.
selected_data["BRCA"] = y_train.loc[selected_data.index].astype(int)


# ============================================================
# DISCRETIZATION
# ============================================================

print("\nDiscretizing gene-expression values...")

for gene in selected_genes:
    # qcut can produce fewer than 3 bins when values are duplicated.
    binned = pd.qcut(
        selected_data[gene],
        q=3,
        labels=False,
        duplicates="drop",
    )

    # Convert to integer categories while handling any possible missing bins.
    if binned.isna().any():
        binned = binned.fillna(binned.mode().iloc[0])

    selected_data[gene] = binned.astype(int)

selected_data["BRCA"] = selected_data["BRCA"].astype(int)

print("\nDiscretized data:")
print(selected_data.head())
print("\nData types:")
print(selected_data.dtypes)


# ============================================================
# PC STRUCTURE LEARNING
# ============================================================

print("\n" + "=" * 60)
print("PC STRUCTURE LEARNING")
print("=" * 60)

pc = PC(data=selected_data)

# Request a DAG rather than a partially directed graph. The DAG returned
# by PC is one compatible orientation; it does not prove causal direction.
estimated_model = pc.estimate(
    variant="stable",
    ci_test="chi_square",
    significance_level=0.05,
    return_type="dag",
)

print("\nLearned network edges:")
print(list(estimated_model.edges()))
print("\nNumber of nodes:", len(estimated_model.nodes()))
print("Number of edges:", len(estimated_model.edges()))


# ============================================================
# BUILD BAYESIAN NETWORK
# ============================================================

print("\n" + "=" * 60)
print("BUILDING BAYESIAN NETWORK")
print("=" * 60)

model = DiscreteBayesianNetwork(list(estimated_model.edges()))
model.add_nodes_from(selected_data.columns)

if not nx.is_directed_acyclic_graph(model):
    raise ValueError(
        "The learned structure contains a directed cycle; cannot build a Bayesian network."
    )

print("\nNetwork created successfully.")
print("Nodes:", list(model.nodes()))
print("Edges:", list(model.edges()))


# ============================================================
# LEARN CPDs
# ============================================================

print("\nLearning Conditional Probability Distributions (CPDs)...")

estimator = BayesianEstimator(model, selected_data)
cpds = estimator.get_parameters(
    prior_type="BDeu",
    equivalent_sample_size=10,
)
model.add_cpds(*cpds)

print("\nCPDs learned successfully.")


# ============================================================
# CHECK BAYESIAN NETWORK
# ============================================================

print("\nChecking Bayesian Network...")
if model.check_model():
    print("Bayesian Network is valid.")
else:
    raise ValueError("Bayesian Network validation failed.")


# ============================================================
# DISPLAY CPDs
# ============================================================

print("\n" + "=" * 60)
print("CONDITIONAL PROBABILITY DISTRIBUTIONS")
print("=" * 60)

for cpd in model.get_cpds():
    print("\n")
    print(cpd)


# ============================================================
# PROBABILISTIC INFERENCE
# ============================================================

print("\n" + "=" * 60)
print("PROBABILISTIC INFERENCE")
print("=" * 60)

inference = VariableElimination(model)

result = inference.query(
    variables=["BRCA"],
    show_progress=False,
)

print("\nP(BRCA):")
print(result)


# ============================================================
# INFERENCE USING GENE EVIDENCE
# ============================================================

# Use the most frequent/highest discretized category observed for each
# of the first two selected genes, rather than assuming category 2 exists.
gene_1 = selected_genes[0]
gene_2 = selected_genes[1] if len(selected_genes) > 1 else selected_genes[0]

evidence = {
    gene_1: int(selected_data[gene_1].max()),
    gene_2: int(selected_data[gene_2].max()),
}

print("\nEvidence:")
print(evidence)

posterior = inference.query(
    variables=["BRCA"],
    evidence=evidence,
    show_progress=False,
)

print("\nP(BRCA | gene evidence):")
print(posterior)


# ============================================================
# SAVE NETWORK EDGES
# ============================================================

edges_df = pd.DataFrame(list(model.edges()), columns=["Source", "Target"])
edges_path = os.path.join(RESULTS_DIR, "bayesian_network_edges.csv")
edges_df.to_csv(edges_path, index=False)

print("\nNetwork edges saved to:")
print(edges_path)


# ============================================================
# SAVE SELECTED GENES
# ============================================================

genes_df = pd.DataFrame({"Selected_Gene": list(selected_genes)})
genes_path = os.path.join(RESULTS_DIR, "bayesian_network_selected_genes.csv")
genes_df.to_csv(genes_path, index=False)

print("\nSelected genes saved to:")
print(genes_path)


# ============================================================
# SAVE CPDS
# ============================================================

cpd_text_path = os.path.join(RESULTS_DIR, "bayesian_network_cpds.txt")
with open(cpd_text_path, "w", encoding="utf-8") as file:
    for cpd in model.get_cpds():
        file.write(str(cpd))
        file.write("\n\n")

print("\nCPDs saved to:")
print(cpd_text_path)


# ============================================================
# SAVE INFERENCE RESULTS
# ============================================================

inference_path = os.path.join(RESULTS_DIR, "bayesian_inference.txt")
with open(inference_path, "w", encoding="utf-8") as file:
    file.write("Prior probability P(BRCA):\n")
    file.write(str(result))
    file.write("\n\n")
    file.write("Evidence:\n")
    file.write(str(evidence))
    file.write("\n\n")
    file.write("Posterior probability P(BRCA | gene evidence):\n")
    file.write(str(posterior))
    file.write("\n")

print("\nInference results saved to:")
print(inference_path)


# ============================================================
# BAYESIAN NETWORK VISUALIZATION
# ============================================================

print("\nCreating Bayesian Network visualization...")

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(model, seed=RANDOM_STATE)

nx.draw_networkx_nodes(model, pos, node_size=3000)
nx.draw_networkx_edges(
    model,
    pos,
    arrows=False,
    width=2
)
nx.draw_networkx_labels(model, pos, font_size=10, font_weight="bold")

plt.title("Bayesian Network - BRCA Gene Expression Analysis", fontsize=14)
plt.axis("off")

network_path = os.path.join(RESULTS_DIR, "bayesian_network.png")
plt.savefig(network_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nNetwork visualization saved to:")
print(network_path)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("BAYESIAN NETWORK ANALYSIS COMPLETED")
print("=" * 60)

print("\nSelected genes:")
for gene in selected_genes:
    print("-", gene)

print("\nNodes:", len(model.nodes()))
print("Edges:", len(model.edges()))
print("\nResults saved in:")
print(RESULTS_DIR)
print("\nCompleted successfully!")
