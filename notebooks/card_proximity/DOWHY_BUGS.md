# DoWhy 0.14 Bugs and Limitations

This document records bugs and limitations discovered while upgrading the Card (1995) notebook from DoWhy 0.12 to DoWhy 0.14.

## Bugs Fixed in 0.14

### ✅ `LinearRegressionEstimator` params indexing bug (Fixed)

**DoWhy 0.12 issue**: The `RegressionEstimator.estimate_effect()` method tried to access `self.model.params[0]` to get the intercept, which failed with statsmodels' parameter Series that uses named indices.

**DoWhy 0.14 fix**: Now uses `self.model.params.iloc[0]` at line 131 of `regression_estimator.py`, which correctly accesses the first parameter by position.

**Impact**: Can now use `model.estimate_effect(estimand, method_name="backdoor.linear_regression")` directly without manual OLS workarounds.

---

## Bugs Still Present in 0.14

### ❌ `LinearSensitivityAnalyzer` scalar iteration bug

**Location**: `dowhy/causal_refuters/linear_sensitivity_analyzer.py:269`

**Symptom**: Calling `add_unobserved_common_cause` with `simulation_method="linear-partial-R2"` raises:
```
TypeError: 'numpy.float64' object is not iterable
```

**Root cause**: At line 268, `self.r2tu_w` is assigned a scalar value when `self.frac_strength_treatment` is scalar (the default is `1.0`):
```python
self.r2tu_w = self.frac_strength_treatment * (r2twj_w / (1 - r2twj_w))
```

Then at line 269, the code tries to iterate:
```python
if any(val >= 1 for val in self.r2tu_w):  # Crashes if r2tu_w is scalar!
    raise ValueError("r2tu_w can not be >= 1...")
```

**Workaround attempted**: Passing `frac_strength_treatment=np.array([1.0])` doesn't help because the assignment at line 268 happens inside a per-benchmark loop that overwrites the variable each iteration.

**Status**: No working workaround found. The `linear-partial-R2` sensitivity method is unusable in 0.14.

**Recommendation**: Use `simulation_method="direct-simulation"` or `simulation_method="e-value"` instead.

---

## Design Limitations

### ⚠️ IV estimator lacks covariate adjustment

**Location**: `dowhy/causal_estimators/instrumental_variable_estimator.py:135-163`

**Issue**: The `InstrumentalVariableEstimator` implements three approaches:

1. **Binary instrument** (lines 140-148): Wald estimator = `(E[Y|Z=1] - E[Y|Z=0]) / (E[X|Z=1] - E[X|Z=0])`
2. **Continuous instrument** (lines 150-153): Covariance ratio = `Cov(Y,Z) / Cov(X,Z)`
3. **Multiple instruments** (lines 155-163): Calls `IV2SLS(outcome, treatment, instruments)` but **no exogenous covariates**

**Problem**: In real applications like Card (1995), you have:
- **Observed confounders**: age, race, region, family background → need covariate adjustment
- **Unobserved confounders**: ability, motivation → need IV

Proper 2SLS should be:
```python
# Stage 1: treatment ~ instruments + observed_confounders
# Stage 2: outcome ~ treatment_hat + observed_confounders
```

DoWhy's IV estimator does:
```python
# Simple ratio: Cov(outcome, instrument) / Cov(treatment, instrument)
# OR for multiple IVs: IV2SLS(outcome, treatment, instruments) with no controls
```

**Workaround**: Implement manual 2SLS using statsmodels:
```python
# Stage 1
X_stage1 = sm.add_constant(pd.concat([instruments, controls], axis=1))
stage1 = sm.OLS(treatment, X_stage1).fit()
treatment_hat = stage1.fittedvalues

# Stage 2
X_stage2 = sm.add_constant(pd.concat([treatment_hat, controls], axis=1))
stage2 = sm.OLS(outcome, X_stage2).fit()
```

For production use, consider the `linearmodels` package which has proper `IV2SLS` with robust standard errors and covariance correction.

**Status**: Design limitation, not a bug. DoWhy's IV is suitable for pedagogical examples with no observed confounders, but not for realistic econometric applications.

---

### ⚠️ `identify_effect` may choose unexpected instruments

**Location**: `dowhy/causal_identifier/identified_estimand.py` (IV identification logic)

**Issue**: With the Card (1995) DAG, DoWhy's `identify_effect()` identifies `momdad14` (lived with both parents at age 14) as an instrumental variable in the IV estimand, rather than the intended `nearc4` (proximity to 4-year college).

**Reason**: From the graph structure, DoWhy considers any variable that:
1. Is an ancestor of treatment (`educ`)
2. Has no direct path to outcome (`lwage`) except through treatment
3. Is not a descendant of a confounder

Both `momdad14` and `nearc4` satisfy these conditions in our DAG.

**Workaround**: Explicitly specify the instrument when calling `estimate_effect()`:
```python
iv_estimate = model.estimate_effect(
    identified_estimand,
    method_name="iv.instrumental_variable",
    method_params={"iv_instrument_name": "nearc4"}  # Explicit instrument selection
)
```

**Status**: Working as designed. DoWhy identifies all valid IVs; the user must select which to use.

---

### ⚠️ Effect modifiers block sensitivity analysis

**Location**: Various sensitivity analyzers (e.g., `linear_sensitivity_analyzer.py:655`)

**Issue**: When estimating with effect modifiers (heterogeneous treatment effects), the sensitivity methods raise:
```
NotImplementedError: The current implementation does not support effect modifiers
```

**Root cause**: At lines 655 and 769 of `add_unobserved_common_cause.py`:
```python
if estimate.estimator._effect_modifier_names is not None and len(estimate.estimator._effect_modifier_names) > 0:
    raise NotImplementedError("The current implementation does not support effect modifiers")
```

DoWhy may auto-detect effect modifiers from the causal graph, even when the user doesn't explicitly request heterogeneous effects.

**Workaround**: Explicitly pass empty effect modifiers:
```python
model = CausalModel(
    data=data, treatment="x", outcome="y", graph=dag,
    effect_modifiers=[]  # Disable auto-detection
)
estimate = model.estimate_effect(
    estimand, method_name="backdoor.linear_regression",
    effect_modifiers=[]  # Ensure no effect modifiers
)
```

**Status**: Design limitation. Sensitivity analysis for heterogeneous treatment effects is not yet implemented.

---

## Working Features in 0.14

### ✅ Direct simulation sensitivity analysis

Works correctly. Use a grid of confounder strengths to visualize sensitivity:

```python
refutation = model.refute_estimate(
    estimand, estimate,
    method_name="add_unobserved_common_cause",
    simulation_method="direct-simulation",
    confounders_effect_on_treatment="linear",
    confounders_effect_on_outcome="linear",
    effect_strength_on_treatment=np.arange(0.0, 0.5, 0.1),
    effect_strength_on_outcome=np.arange(0.0, 0.5, 0.1),
)
```

Produces a contour plot showing the range of new estimates under different confounding scenarios.

### ✅ E-value sensitivity analysis

Works correctly. Computes minimum confounding strength (risk-ratio scale) needed to explain away the result:

```python
refutation = model.refute_estimate(
    estimand, estimate,
    method_name="add_unobserved_common_cause",
    simulation_method="e-value",
)
```

Output includes:
- E-value for point estimate
- E-value for confidence interval
- Benchmarking against observed covariates

### ✅ Standard refutation tests

All work correctly:

- **Placebo treatment**: `method_name="placebo_treatment_refuter"`
- **Random common cause**: `method_name="random_common_cause"`
- **Data subset**: `method_name="data_subset_refuter"`
- **Bootstrap**: `method_name="bootstrap_refuter"`

---

## Summary

**Use DoWhy 0.14 for**:
- Backdoor estimation with linear regression (bug fixed!)
- Standard refutation tests (placebo, random CC, data subset)
- Direct simulation sensitivity analysis
- E-value sensitivity analysis

**Avoid or work around**:
- `linear-partial-R2` sensitivity (broken)
- IV estimation with observed confounders (missing feature — use manual 2SLS)
- Effect modifiers with sensitivity analysis (not supported)

**For production IV work**: Use `linearmodels.iv.IV2SLS` or implement manual 2SLS with proper covariance correction.
