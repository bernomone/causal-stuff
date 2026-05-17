"""Card (1995) proximity-to-college dataset loader with caching.

This module provides access to the Card (1995) dataset used in "Using Geographic
Variation in College Proximity to Estimate the Return to Schooling" (NBER WP 4483).
The dataset contains 3,010 men from the National Longitudinal Survey of Young Men
with data on education, wages, and geographic proximity to colleges.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class CardDataset:
    """
    Loader for the Card (1995) proximity-to-college dataset.

    This dataset is used to estimate the causal effect of education on wages
    using geographic proximity to college as an instrumental variable. The
    data comes from the NLSYM and is available via statsmodels.

    Reference
    ---------
    Card, D. (1995). "Using Geographic Variation in College Proximity to Estimate
    the Return to Schooling." NBER Working Paper No. 4483.
    https://www.nber.org/papers/w4483

    Examples
    --------
    >>> ds = CardDataset()
    >>> data = ds.data  # 3010 rows, all columns
    >>> dag_data = ds.complete_family[ds.DAG_COLUMNS]  # ~2215 rows, no missing
    """

    # Column groups for easy reference
    INSTRUMENTS = ["nearc2", "nearc4"]
    TREATMENT = "educ"
    OUTCOME = "lwage"
    DAG_COLUMNS = [
        "nearc2",
        "nearc4",
        "educ",
        "lwage",
        "age",
        "black",
        "married",
        "south",
        "smsa",
        "fatheduc",
        "motheduc",
        "momdad14",
    ]

    # Columns excluded from analysis (deterministic or redundant)
    EXCLUDED_DETERMINISTIC = ["id", "weight", "wage", "expersq"]

    def __init__(self, cache_dir: Path | str | None = None):
        """
        Initialize the Card dataset loader.

        Parameters
        ----------
        cache_dir : Path, str, or None
            Directory for caching the dataset.
            Defaults to <project_root>/data/card/
        """
        if cache_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            cache_dir = project_root / "data" / "card"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        """Load the dataset from cache or download from statsmodels."""
        if self._df is not None:
            return self._df

        cache_path = self._cache_dir / "card.pkl"

        if cache_path.exists():
            self._df = pd.read_pickle(cache_path)
        else:
            # Load from statsmodels
            import statsmodels.datasets

            data = statsmodels.datasets.get_rdataset("card", "wooldridge")
            df = data.data

            # Drop excluded columns
            df = df.drop(columns=self.EXCLUDED_DETERMINISTIC, errors="ignore")

            # Cache to disk
            df.to_pickle(cache_path)
            self._df = df

        return self._df

    @property
    def data(self) -> pd.DataFrame:
        """
        Full dataset with all 3,010 observations.

        Some columns have missing values:
        - fatheduc: ~795 missing
        - motheduc: ~795 missing
        - married: ~7 missing
        - libcrd14: ~53 missing
        """
        return self._load()

    @property
    def complete_family(self) -> pd.DataFrame:
        """
        Subset with complete cases on family background variables.

        Drops rows with missing values in: fatheduc, motheduc, married.
        Result: ~2,215 observations with complete data on all DAG columns.

        This is the recommended dataset for DAG falsification analyses that
        include parental education as confounders.
        """
        df = self.data
        return df.dropna(subset=["fatheduc", "motheduc", "married"])
