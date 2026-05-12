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
│   └── lalonde.py     # LaLonde/Dehejia-Wahba NSW dataset
├── methods/            # Reusable matching/estimation implementations
│   └── matching.py    # MahalanobisMatch, PropensityScoreMatch classes
├── notebooks/          # Experimental notebooks (numbered by workflow)
├── data/               # Cached dataset files (git-ignored)
└── pyproject.toml      # Dependencies and project metadata
```

## Architecture & Design

### Dataset Loading Pattern

The `datasets/` module provides cached loaders that download data from canonical sources on first access and store locally in `data/`. Each dataset loader exposes:
- **Experimental data**: For computing true treatment effects (ground truth)
- **Observational data**: Semi-observational variants with selection bias for testing methods

Example: `LalondeDataset` provides `.experimental` (RCT data) and `.observational(control=ControlGroup.PSID)` (experimental treated + non-experimental control).

### Matching Methods Module

`methods/matching.py` contains scikit-learn-style matching estimators:
- **MahalanobisMatch**: Nearest-neighbor matching using Mahalanobis distance (accounts for covariate correlations)
- **PropensityScoreMatch**: Nearest-neighbor matching on propensity scores

Both follow a `.match()` → `.estimate_att()` API pattern.

### Notebook Workflow

Notebooks are numbered sequentially by analysis flow:
1. `01_lalonde_psm.ipynb` - Propensity score matching baseline
2. `02_lalonde_covariate_matching.ipynb` - Direct covariate matching (Mahalanobis)
3. `03_lalonde_metalearners.ipynb` - Metalearner comparison (S/T/X-learners)

Each notebook:
- Loads data via `datasets.lalonde.LalondeDataset`
- Estimates ATT using one or more methods
- Compares against `ds.true_att` to evaluate bias

## Testing Causal Methods

When adding new methods or datasets:

1. **Start with experimental data** (`ds.experimental`) to verify the method recovers the true effect without selection bias
2. **Test on observational data** (`ds.observational()`) to assess robustness to confounding
3. **Compare multiple control groups**: PSID, PSID2, PSID3, CPS variants have different degrees of covariate imbalance
4. **Report bias**: `estimated_att - ds.true_att`

## Common Covariates (LaLonde)

Standard covariate set for matching:
```python
covariates = ['age', 'educ', 'black', 'hisp', 'married', 'nodegr', 're74', 're75', 'u74', 'u75']
```
- `re74`, `re75`: Real earnings in 1974, 1975 (pre-treatment)
- `u74`, `u75`: Unemployment indicators (derived: 1 if earnings == 0)
- `re78`: Outcome variable (1978 earnings)
- `treat`: Treatment indicator (job training program)

## Next Steps (from description.md)

Planned extensions:
1. Implement Gower distance for mixed-type covariate matching
2. Add datasets from CauSciBench (https://github.com/causalNLP/CauSciBench)
3. Integrate causal discovery: learn graph → estimate ATT with DoWhy
