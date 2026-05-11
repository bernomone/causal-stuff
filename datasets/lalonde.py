"""LaLonde / Dehejia-Wahba NSW dataset loader with caching.

This module provides access to both the experimental LaLonde data and
non-experimental comparison groups (PSID, CPS) for testing causal inference
methods on observational data with selection bias.
"""

from __future__ import annotations

import urllib.request
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


class ControlGroup(Enum):
    """Available non-experimental control groups from NBER."""

    PSID = "psid_controls"  # 2,490 obs
    PSID2 = "psid2_controls"  # 253 obs
    PSID3 = "psid3_controls"  # 128 obs
    CPS = "cps_controls"  # 15,992 obs
    CPS2 = "cps2_controls"  # 2,369 obs
    CPS3 = "cps3_controls"  # 429 obs


_URLS = {
    "nswre74_treated": "https://www.nber.org/~rdehejia/data/nswre74_treated.txt",
    "nswre74_control": "https://www.nber.org/~rdehejia/data/nswre74_control.txt",
    "psid_controls": "https://www.nber.org/~rdehejia/data/psid_controls.txt",
    "psid2_controls": "https://www.nber.org/~rdehejia/data/psid2_controls.txt",
    "psid3_controls": "https://www.nber.org/~rdehejia/data/psid3_controls.txt",
    "cps_controls": "https://www.nber.org/~rdehejia/data/cps_controls.txt",
    "cps2_controls": "https://www.nber.org/~rdehejia/data/cps2_controls.txt",
    "cps3_controls": "https://www.nber.org/~rdehejia/data/cps3_controls.txt",
}

COLUMNS = [
    "treat",
    "age",
    "educ",
    "black",
    "hisp",
    "married",
    "nodegr",
    "re74",
    "re75",
    "re78",
]


class LalondeDataset:
    """
    Loader for the LaLonde / Dehejia-Wahba NSW dataset variants.

    Provides both the experimental sample (for computing the true ATE/ATT)
    and semi-observational datasets (experimental treated + non-experimental
    controls) for testing causal inference methods.

    Examples
    --------
    >>> ds = LalondeDataset()
    >>> experimental_data = ds.experimental  # 445 rows (RCT)
    >>> observational_data = ds.observational()  # 2675 rows (with PSID)
    >>> print(f"True ATT: ${ds.true_att:.2f}")
    """

    def __init__(self, cache_dir: Path | str | None = None):
        """
        Initialize the LaLonde dataset loader.

        Parameters
        ----------
        cache_dir : Path, str, or None
            Directory for caching downloaded files.
            Defaults to <project_root>/data/lalonde/
        """
        if cache_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            cache_dir = project_root / "data" / "lalonde"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._df_cache: dict[str, pd.DataFrame] = {}

    def _download(self, name: str) -> Path:
        """Download a file from NBER if not cached."""
        path = self._cache_dir / f"{name}.txt"
        if not path.exists():
            url = _URLS[name]
            urllib.request.urlretrieve(url, path)
        return path

    def _load(self, name: str) -> pd.DataFrame:
        """Load a data file into a DataFrame with standard processing."""
        if name not in self._df_cache:
            path = self._download(name)
            df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
            df = df.astype({"treat": "bool"}, copy=False)
            df["u74"] = np.where(df["re74"] == 0, 1.0, 0.0)
            df["u75"] = np.where(df["re75"] == 0, 1.0, 0.0)
            self._df_cache[name] = df
        return self._df_cache[name]

    @property
    def treated(self) -> pd.DataFrame:
        """Experimental treated group (185 rows)."""
        return self._load("nswre74_treated")

    @property
    def experimental_control(self) -> pd.DataFrame:
        """Experimental control group (260 rows)."""
        return self._load("nswre74_control")

    @property
    def experimental(self) -> pd.DataFrame:
        """Full experimental sample: 185 treated + 260 control = 445 rows."""
        return pd.concat(
            [self.treated, self.experimental_control], ignore_index=True
        )

    def control_group(self, group: ControlGroup) -> pd.DataFrame:
        """
        Load a specific non-experimental control group.

        Parameters
        ----------
        group : ControlGroup
            Which control group to load.

        Returns
        -------
        pd.DataFrame
            Control group data with standard column names.
        """
        return self._load(group.value)

    def observational(
        self, control: ControlGroup = ControlGroup.PSID
    ) -> pd.DataFrame:
        """
        Semi-observational dataset: experimental treated + non-experimental control.

        This creates a dataset with selection bias, suitable for testing
        whether causal inference methods can recover the true ATT.

        Parameters
        ----------
        control : ControlGroup, default=ControlGroup.PSID
            Which non-experimental comparison group to use.
            Default is PSID (2,490 obs), most commonly used in the literature.

        Returns
        -------
        pd.DataFrame
            Combined dataset with columns matching DoWhy convention.
        """
        return pd.concat(
            [self.treated, self.control_group(control)], ignore_index=True
        )

    @property
    def true_ate(self) -> float:
        """
        True experimental ATE from the randomized trial.

        Computed as mean(re78|treated) - mean(re78|control) from
        the experimental sample.
        """
        exp = self.experimental
        treated_mean = exp.loc[exp["treat"], "re78"].mean()
        control_mean = exp.loc[~exp["treat"], "re78"].mean()
        return treated_mean - control_mean

    @property
    def true_att(self) -> float:
        """
        True experimental ATT.

        In a properly randomized experiment, ATT == ATE.
        Provided for semantic clarity when comparing observational estimates.
        """
        return self.true_ate
