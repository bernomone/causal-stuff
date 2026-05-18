"""
This script defines the Directed Acyclic Graph (DAG) for the
falsification and instrumental variable (IV) estimation strategies
used in Card's (1995) analysis of the impact of college proximity
on educational attainment.

Reference: Card, D. (1995). "Using Geographic Variation in College Proximity to Estimate the Return to Schooling." NBER Working Paper No. 4483. https://www.nber.org/papers/w4483

Dataset: National Longitudinal Survey of Young Men (NLSYM), N=3,010, with data on education, wages, family background, and geographic proximity to 2-year and 4-year colleges.

DAG Column Definitions (12 variables used, ~2,215 complete cases):

Instruments:
    - nearc2: Binary indicator for proximity to 2-year college in county of residence at age 14
    - nearc4: Binary indicator for proximity to 4-year college in county of residence at age 14

Treatment & Outcome:
    - educ: Years of education completed
    - lwage: Log of hourly wage in 1976

Demographics:
    - age: Age in years (range: 24-34)
    - black: Binary indicator for Black race
    - married: Marital status code (1=single, 2-6 represent different married/widowed states)

Geography:
    - south: Binary indicator for residence in southern region at age 14
    - smsa: Binary indicator for residence in Standard Metropolitan Statistical Area (urban)

Family Background:
    - fatheduc: Father's years of education (0-18)
    - motheduc: Mother's years of education (0-18)
    - momdad14: Binary indicator for living with both parents at age 14

Excluded columns (deterministic or redundant):
    - id: Individual identifier
    - weight: Sample weight
    - wage: Hourly wage in levels (use lwage instead for log-linear models)
    - expersq: Experience squared (redundant given age and educ)
    - libcrd14: Library card at age 14 (high missingness ~53, not in core DAG)
"""


import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import networkx as nx
import matplotlib.pyplot as plt
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm.falsify import apply_suggestions
from dowhy.gcm.independence_test import approx_kernel_based
from datasets import CardDataset

# Load Card dataset
ds = CardDataset()

# Use complete_family: drops rows with missing fatheduc, motheduc, married
# This gives ~2,215 complete cases on all DAG variables
data = ds.complete_family.reset_index(drop=True)

print(f"Full dataset: {ds.data.shape}")
print(f"Complete cases (family vars): {data.shape}")
print(f"\nMissing values in key columns:")
print(data[ds.DAG_COLUMNS].isnull().sum())
print(f"\nDAG columns (12): {ds.DAG_COLUMNS}")

# Subset to DAG columns only
dag_data = data[ds.DAG_COLUMNS].copy()

print(f"DAG data shape: {dag_data.shape}")
print(f"\nSummary statistics:\n {dag_data.describe()}")


# Build the causal DAG
dag = nx.DiGraph()

edges = [
    # Family background affects education and wages
    ('fatheduc', 'educ'),
    ('motheduc', 'educ'),
    ('momdad14', 'educ'),
    ('fatheduc', 'lwage'),
    ('motheduc', 'lwage'),

    # Family background affects geographic location (where they lived at age 14)
    # More educated families may select into areas with colleges
    ('fatheduc', 'nearc2'),
    ('fatheduc', 'nearc4'),
    ('motheduc', 'nearc2'),
    ('motheduc', 'nearc4'),

    # Geography affects college proximity
    ('south', 'nearc2'),
    ('south', 'nearc4'),
    ('smsa', 'nearc2'),
    ('smsa', 'nearc4'),

    # Geography affects education opportunities and wages
    ('south', 'educ'),
    ('south', 'lwage'),
    ('smsa', 'educ'),
    ('smsa', 'lwage'),

    # Race affects education and wages (discrimination, resources)
    ('black', 'educ'),
    ('black', 'lwage'),

    # IV: College proximity affects education (relevance condition)
    ('nearc2', 'educ'),
    ('nearc4', 'educ'),

    # KEY: NO direct edges from nearc2/nearc4 to lwage (exclusion restriction)
    # Proximity only affects wages through education

    # Education affects wages (treatment effect of interest)
    ('educ', 'lwage'),

    # Education affects marriage
    ('educ', 'married'),

    # Marriage affects wages (marriage premium)
    ('married', 'lwage'),

    # Age affects wages (labor market experience)
    ('age', 'lwage'),
]

dag.add_edges_from(edges)

print(f"Proposed DAG:")
print(f"  Nodes: {dag.number_of_nodes()}")
print(f"  Edges: {dag.number_of_edges()}")
print(f"  Is DAG: {nx.is_directed_acyclic_graph(dag)}")
print(f"\nNodes: {sorted(dag.nodes())}")
print(f"\nIV check for nearc4:")
print(f"  - Affects educ? {nx.has_path(dag, 'nearc4', 'educ')}")
print(f"  - Direct path to lwage (excluding through educ)? ", end="")
# Check if there's a path from nearc4 to lwage that doesn't go through educ
dag_no_educ = dag.copy()
dag_no_educ.remove_node('educ')
has_direct_path = nx.has_path(dag_no_educ, 'nearc4', 'lwage')
print(f"{has_direct_path} (should be False for valid IV)")



# Run falsification test (this may take 20-30 minutes with 12 nodes and 2215 rows)
result = falsify_graph(
    dag,
    dag_data,
    conditional_independence_test=approx_kernel_based,  # MUCH faster than kernel
    n_jobs=10, 
    show_progress_bar=True,
    suggestions=True,  # Enable causal minimality testing
)
print(result)

# Apply the suggested corrections
corrected_dag = apply_suggestions(dag, result)

print(f"Original DAG: {dag.number_of_edges()} edges")
print(f"Corrected DAG: {corrected_dag.number_of_edges()} edges")
print(f"Edges removed: {dag.number_of_edges() - corrected_dag.number_of_edges()}")

# Show which edges were removed
removed_edges = set(dag.edges()) - set(corrected_dag.edges())
if removed_edges:
    print(f"\nRemoved edges (violate causal minimality):")
    for edge in sorted(removed_edges):
        print(f"  {edge[0]} → {edge[1]}")
else:
    print("\nNo edges removed - all edges are necessary.")

# Test the corrected DAG
if corrected_dag.number_of_edges() < dag.number_of_edges():
    print("Testing the corrected DAG...\n")
    corrected_result = falsify_graph(
        corrected_dag,
        dag_data,
        n_jobs=10,
        show_progress_bar=True,
    )
    print(corrected_result)
    
else:
    print("\nThe corrected DAG is identical to the original - no redundant edges found.")
    print("The falsification may be due to missing edges rather than spurious ones.")


# Save DAGs to data/
output_dir = project_root / "data" / "card"
output_dir.mkdir(parents=True, exist_ok=True)

# Save as GraphML (human-readable XML format)
dag_path = output_dir / "card_dag_original.graphml"
corrected_dag_path = output_dir / "card_dag_corrected.graphml"

nx.write_graphml(dag, dag_path)
nx.write_graphml(corrected_dag, corrected_dag_path)

print(f"\nDAGs saved:")
print(f"  Original: {dag_path}")
print(f"  Corrected: {corrected_dag_path}")


# Create visualizations
def visualize_dag(graph, title, output_path):
    """Create a hierarchical visualization of the DAG."""
    fig, ax = plt.subplots(figsize=(16, 12))

    # Use hierarchical layout (by topological sort layers)
    pos = nx.spring_layout(graph, seed=42, k=2, iterations=50)

    # Identify node types for color coding
    instruments = ['nearc2', 'nearc4']
    treatment = ['educ']
    outcome = ['lwage']
    confounders = [n for n in graph.nodes() if n not in instruments + treatment + outcome]

    # Draw nodes by type
    nx.draw_networkx_nodes(graph, pos, nodelist=instruments,
                          node_color='lightgreen', node_size=2000,
                          label='Instruments', ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=treatment,
                          node_color='lightblue', node_size=2000,
                          label='Treatment', ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=outcome,
                          node_color='salmon', node_size=2000,
                          label='Outcome', ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=confounders,
                          node_color='lightgray', node_size=2000,
                          label='Confounders', ax=ax)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color='black',
                          arrows=True, arrowsize=20,
                          arrowstyle='->', width=1.5,
                          connectionstyle='arc3,rad=0.1', ax=ax)

    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold', ax=ax)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(scatterpoints=1, loc='upper left', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {output_path}")


# Generate visualizations
print("\nGenerating visualizations...")
visualize_dag(dag,
              f"Original DAG ({dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges)",
              output_dir / "card_dag_original.png")

visualize_dag(corrected_dag,
              f"Corrected DAG ({corrected_dag.number_of_nodes()} nodes, {corrected_dag.number_of_edges()} edges)",
              output_dir / "card_dag_corrected.png")

print("\nAll DAG artifacts saved to:", output_dir)