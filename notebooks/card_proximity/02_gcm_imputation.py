"""
This script performs imputation of missing values using Graphical Causal Models (GCM) on the Card dataset. We use the complete cases from the family background variables to build a causal DAG and then apply GCM-based imputation to fill in missing values in the dataset. The imputed dataset can then be used for further analysis, such as estimating the effect of college proximity on education and wages while accounting for confounding factors.
"""

import gc
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import networkx as nx
import dowhy.gcm as gcm
from datasets import CardDataset
from lightgbm import LGBMRegressor

# Load Card dataset
ds = CardDataset()
data = ds.data[ds.DAG_COLUMNS].reset_index(drop=True)

# Use complete_family: drops rows with missing fatheduc, motheduc, married
# This gives ~2,215 complete cases on all DAG variables
data_nonnull = ds.complete_family[ds.DAG_COLUMNS].reset_index(drop=True)

print(f"Full dataset: {data.shape}")
print(f"Complete cases (family vars): {data_nonnull.shape}")

tot = len(data)
print(f"\nMissing values in key columns:")
for col in ds.DAG_COLUMNS:
    n_missing = data[col].isnull().sum()
    pct_missing = n_missing / tot * 100
    if n_missing > 0:
        print(f"{col}: {n_missing} missing ({pct_missing:.1f}%)")

# Define GCM structure
dag = nx.read_graphml(project_root / "data" / "card" / "card_dag_original.graphml")
causal_model = gcm.StructuralCausalModel(dag)

# Start with auto-assignment
gcm.auto.assign_causal_mechanisms(causal_model, data_nonnull, quality=gcm.auto.AssignmentQuality.GOOD)

print("Fitting causal model with custom mechanisms...")
gcm_eval_summary = gcm.fit(causal_model, data_nonnull, return_evaluation_summary=True)
print(gcm_eval_summary)

# Impute missing values using the fitted causal model
print("\n" + "="*60)
print("Imputing missing values using GCM...")
print("="*60)

import pandas as pd
import numpy as np

# Manual imputation using causal mechanisms
imputed_data = data.copy()

# Get topological order of nodes (process in causal order)
topo_order = list(nx.topological_sort(dag))

# For each row with missing values, impute in topological order
rows_with_missing = data.isnull().any(axis=1)
n_rows_to_impute = rows_with_missing.sum()
print(f"Rows with missing values: {n_rows_to_impute} / {len(data)}")

for idx in data[rows_with_missing].index:
    if idx % 100 == 0:
        print(f"  Imputing row {idx} / {len(data)}")

    # Process nodes in topological order
    for node in topo_order:
        if pd.isnull(imputed_data.loc[idx, node]):
            # Get the causal mechanism for this node
            mechanism = causal_model.causal_mechanism(node)

            # Get parent values
            parents = list(dag.predecessors(node))

            if len(parents) == 0:
                # Root node: sample from marginal
                try:
                    sample = mechanism.draw_samples(1)
                    imputed_data.loc[idx, node] = sample[0] if isinstance(sample, np.ndarray) else sample
                except Exception as e:
                    print(f"Warning: Could not impute root node {node} at row {idx}: {e}")
            else:
                # Non-root: sample conditioned on parents
                try:
                    parent_values = imputed_data.loc[idx, parents].values.reshape(1, -1)
                    sample = mechanism.draw_samples(parent_values)
                    imputed_data.loc[idx, node] = sample[0] if isinstance(sample, np.ndarray) else sample
                except Exception as e:
                    print(f"Warning: Could not impute {node} at row {idx}: {e}")

print("Imputation complete!")

# Verify imputation worked
print(f"\nMissing values after imputation:")
n_still_missing = 0
for col in ds.DAG_COLUMNS:
    n_missing = imputed_data[col].isnull().sum()
    if n_missing > 0:
        print(f"  {col}: {n_missing} missing")
        n_still_missing += n_missing
    else:
        print(f"  {col}: complete ✓")

print(f"\nOriginal data shape: {data.shape}")
print(f"Imputed data shape: {imputed_data.shape}")
print(f"Total values imputed: {data.isnull().sum().sum() - n_still_missing}")

# Evaluate the imputed data quality
print("\n" + "="*60)
print("Evaluating imputed data quality...")
print("="*60)

generated_data = gcm.draw_samples(causal_model, num_samples=3000)
kl_div_complete = gcm.divergence.auto_estimate_kl_divergence(data_nonnull.to_numpy(), generated_data.to_numpy())
print(f"KL divergence (complete cases vs generated): {kl_div_complete:.4f}")

kl_div_imputed = gcm.divergence.auto_estimate_kl_divergence(imputed_data.to_numpy(), generated_data.to_numpy())
print(f"KL divergence (imputed vs generated): {kl_div_imputed:.4f}")