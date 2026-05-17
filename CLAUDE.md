# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a causal inference experimentation repository for testing various causal frameworks and methods. Each test is organized as a Jupyter notebook or Python script, focusing on comparing different causal estimation approaches using benchmark datasets.

**Primary Goal**: Test and compare causal inference methods (matching, double ML, metalearners) on standard datasets to evaluate their performance in recovering true treatment effects.

## Key Frameworks & Dependencies

- **DoWhy**: Causal inference framework with support for various estimation methods
- **EconML**: Microsoft's library for heterogeneous treatment effects and double ML
- **CausalML**: Uber's library for uplift modeling and metalearners
- **causal-learn**: Causal discovery algorithms
- **scikit-learn**: Standard ML models for propensity score estimation
- **LightGBM**: Gradient boosting framework (v4.6.0+)

## Development Commands

**Environment management**:
```bash
# Activate virtual environment (located in .venv/)
source .venv/bin/activate

# Install/sync dependencies (uses uv)
uv sync
```

**Running notebooks**:
```bash
# Start Jupyter Lab
jupyter lab

# Run a specific notebook from CLI
jupyter nbconvert --execute --to notebook notebooks/01_lalonde_psm.ipynb
```

## Repository Structure

```
causal-stuff/
├── datasets/           # Dataset loaders with caching
│   ├── lalonde.py     # LaLonde/Dehejia-Wahba NSW dataset (matching/metalearners)
│   └── card.py        # Card (1995) proximity-to-college dataset (IV estimation)
├── methods/            # Reusable matching/estimation implementations
│   └── matching.py    # MahalanobisMatch, PropensityScoreMatch classes
├── notebooks/          # Experimental notebooks (organized by dataset)
│   ├── lalonde/       # LaLonde dataset experiments (PSM, matching, metalearners)
│   └── card_proximity/# Card dataset experiments (DAG falsification, IV estimation)
├── data/               # Cached dataset files (git-ignored)
└── pyproject.toml      # Dependencies and project metadata
```

## Architecture & Design

### Dataset Loading Pattern

The `datasets/` module provides cached loaders that download data from canonical sources on first access and store locally in `data/`. Dataset loaders follow two patterns:

**1. Experimental/Observational pattern (LaLonde)**:
- **Experimental data**: RCT data for computing true treatment effects (ground truth)
- **Observational data**: Semi-observational variants with selection bias for testing methods
- Example: `LalondeDataset` provides `.experimental` (RCT data) and `.observational(control=ControlGroup.PSID)` (experimental treated + non-experimental control)

**2. Instrumental variable pattern (Card)**:
- **Full data**: All observations with potential missing values in some covariates
- **Complete cases**: Subset with no missing values on key DAG variables
- Example: `CardDataset` provides `.data` (N=3,010) and `.complete_family` (~N=2,215) with complete family background variables
- Exposes column groups: `INSTRUMENTS`, `TREATMENT`, `OUTCOME`, `DAG_COLUMNS`, `EXCLUDED_DETERMINISTIC`

### Matching Methods Module

`methods/matching.py` contains scikit-learn-style matching estimators:
- **MahalanobisMatch**: Nearest-neighbor matching using Mahalanobis distance (accounts for covariate correlations)
- **PropensityScoreMatch**: Nearest-neighbor matching on propensity scores

Both follow a `.match()` → `.estimate_att()` API pattern.

### Notebook Workflow

Notebooks are organized by dataset and numbered sequentially within each directory:

**LaLonde notebooks** (`notebooks/lalonde/`):
1. `01_lalonde_psm.ipynb` - Propensity score matching baseline
2. `02_lalonde_covariate_matching.ipynb` - Direct covariate matching (Mahalanobis)
3. `03_lalonde_metalearners.ipynb` - Metalearner comparison (S/T/X-learners)
4. `04_lalonde_dag_falsification.ipynb` - DAG-based causal discovery
5. `05_lalonde_dowhy_backdoor.ipynb` - DoWhy backdoor estimation
6. `05b_lalonde_dowhy_mahalanobis.ipynb` - DoWhy with Mahalanobis matching

Standard workflow:
- Loads data via `datasets.lalonde.LalondeDataset`
- Estimates ATT using one or more methods
- Compares against `ds.true_att` to evaluate bias

**Card notebooks** (`notebooks/card_proximity/`):
1. `01_card_dag_falsification_iv.ipynb` - DAG falsification + IV estimation

Workflow:
- Proposes an initial DAG (deliberately imperfect with exclusion restriction violations)
- Runs `falsify_graph` to detect LMC violations and get edge removal suggestions
- Applies suggestions to correct the DAG (removes spurious edges, validates IV exclusion restriction)
- Identifies causal effects via both backdoor adjustment and instrumental variables
- Estimates returns to education using OLS and 2SLS, compares to Card (1995) reference results
- Validates with refutation checks

## Testing Causal Methods

When adding new methods or datasets:

1. **Start with experimental data** (`ds.experimental`) to verify the method recovers the true effect without selection bias
2. **Test on observational data** (`ds.observational()`) to assess robustness to confounding
3. **Compare multiple control groups**: PSID, PSID2, PSID3, CPS variants have different degrees of covariate imbalance
4. **Report bias**: `estimated_att - ds.true_att`

## Dataset-Specific Notes

### LaLonde Dataset

Standard covariate set for matching:
```python
covariates = ['age', 'educ', 'black', 'hisp', 'married', 'nodegr', 're74', 're75', 'u74', 'u75']
```
- `re74`, `re75`: Real earnings in 1974, 1975 (pre-treatment)
- `u74`, `u75`: Unemployment indicators (derived: 1 if earnings == 0)
- `re78`: Outcome variable (1978 earnings)
- `treat`: Treatment indicator (job training program)

### Card Dataset

Key variables for IV estimation:
```python
INSTRUMENTS = ["nearc2", "nearc4"]  # Proximity to 2-year/4-year college
TREATMENT = "educ"                  # Years of education
OUTCOME = "lwage"                   # Log wages
```

**DAG columns** (12 variables with complete cases):
- Instruments: `nearc2`, `nearc4`
- Treatment/outcome: `educ`, `lwage`
- Demographics: `age`, `black`, `married`
- Geography: `south`, `smsa` (metropolitan area)
- Family background: `fatheduc`, `motheduc`, `momdad14` (lived with both parents at 14)

**Reference**: Card, D. (1995). "Using Geographic Variation in College Proximity to Estimate the Return to Schooling." NBER Working Paper No. 4483.

**Card's results**: OLS return ~7.3%, IV return ~12-13% (IV > OLS suggests measurement error or LATE for high-return compliers)

## DAG Falsification Workflow (DoWhy)

The Card notebook demonstrates the DAG falsification and causal inference workflow:

1. **Propose**: Construct a causal DAG based on domain knowledge
2. **Falsify**: Use `dowhy.gcm.falsify.falsify_graph` to test Local Markov Conditions via kernel-based conditional independence tests
3. **Identify**: If the DAG is not rejected, use DoWhy's `CausalModel.identify_effect()` to find backdoor and IV estimands
4. **Estimate**: Apply both backdoor (linear regression) and IV (2SLS) methods
5. **Refute**: Validate estimates with robustness checks

Note: If the DAG is rejected by falsification tests, causal minimality suggestions can be obtained by re-running `falsify_graph` with `suggestions=True`, then applying corrections via `apply_suggestions`.

**NetworkX 3.x compatibility**: Current notebooks patch `nx.d_separated` for DoWhy 0.12 compatibility:
```python
nx.algorithms.d_separated = nx.algorithms.d_separation.is_d_separator
nx.d_separated = nx.algorithms.d_separation.is_d_separator
```

## Next Steps

Planned extensions:
1. Implement Gower distance for mixed-type covariate matching
2. Add datasets from CauSciBench (https://github.com/causalNLP/CauSciBench)
3. Extend DAG falsification workflow to LaLonde dataset
