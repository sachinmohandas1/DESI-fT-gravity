"""Gaussian BAO likelihood for DESI cobaya-format data files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from desi_ft.models.base import CosmologicalModel, DEFAULT_RD

# Map from cobaya observable names to the model method that predicts them.
# Some observables are the *inverse* of a standard quantity.
_OBSERVABLE_MAP = {
    "DV_over_rs": ("DV_over_rd", False),
    "DM_over_rs": ("DM_over_rd", False),
    "DH_over_rs": ("DH_over_rd", False),
    "rs_over_DV": ("DV_over_rd", True),   # data = r_d / D_V  →  theory = 1 / (D_V/r_d)
    "DA_over_rs": ("DM_over_rd", False),   # D_A = D_M/(1+z) handled separately
    "Hz_rs":      ("DH_over_rd", True),    # H(z)*r_d = c / (D_H/r_d)  (inverse relation)
}


@dataclass
class BAODataPoint:
    """A single BAO measurement."""
    z_eff: float
    obs_name: str
    value: float


@dataclass
class BAOLikelihood:
    """Gaussian BAO likelihood.

    Reads cobaya-format mean and covariance files and evaluates

        log L = -0.5 * (d - t)^T C^{-1} (d - t)

    Parameters
    ----------
    data_file : str or Path
        Path to the mean-values file (``# z  value  observable``).
    cov_file : str or Path
        Path to the covariance matrix file.
    """

    data_points: list[BAODataPoint] = field(default_factory=list)
    data_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    inv_cov: np.ndarray = field(default_factory=lambda: np.array([]))
    _log_det_term: float = 0.0

    def __init__(self, data_file: str | Path, cov_file: str | Path) -> None:
        self.data_points = self._parse_data(data_file)
        self.data_vector = np.array([dp.value for dp in self.data_points])
        cov = np.loadtxt(cov_file)
        assert cov.shape == (len(self.data_points), len(self.data_points)), (
            f"Covariance shape {cov.shape} does not match "
            f"{len(self.data_points)} data points"
        )
        self.inv_cov = np.linalg.inv(cov)
        # Pre-compute constant normalization (optional, for evidence later)
        sign, logdet = np.linalg.slogdet(cov)
        self._log_det_term = -0.5 * logdet

    @staticmethod
    def _parse_data(path: str | Path) -> list[BAODataPoint]:
        """Parse a cobaya-format BAO mean file.

        Expected format (whitespace-separated)::

            # [z] [value at z] [quantity]
            0.295  7.925  DV_over_rs
        """
        points: list[BAODataPoint] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                z_eff = float(parts[0])
                value = float(parts[1])
                obs_name = parts[2]
                points.append(BAODataPoint(z_eff, obs_name, value))
        return points

    def theory_vector(
        self, model: CosmologicalModel, params: dict[str, float], rd: float = DEFAULT_RD
    ) -> np.ndarray:
        """Compute theory predictions matching the data vector ordering."""
        theory = np.empty(len(self.data_points))
        for i, dp in enumerate(self.data_points):
            if dp.obs_name not in _OBSERVABLE_MAP:
                raise ValueError(f"Unknown BAO observable: {dp.obs_name}")

            method_name, is_inverse = _OBSERVABLE_MAP[dp.obs_name]

            if dp.obs_name == "DA_over_rs":
                # D_A(z) = D_M(z) / (1+z)
                dm_over_rd = model.DM_over_rd(dp.z_eff, params, rd)
                pred = dm_over_rd / (1.0 + dp.z_eff)
            elif dp.obs_name == "Hz_rs":
                # H(z)*r_d = c / (D_H/r_d)  but D_H/r_d = c/(H*r_d)
                # So H*r_d = c / (D_H/r_d) ... wait, D_H = c/H so D_H/r_d = c/(H*r_d)
                # Therefore H*r_d = c / (D_H/r_d) — yes, inverse
                dh_over_rd = model.DH_over_rd(dp.z_eff, params, rd)
                from desi_ft.cosmology.distances import C_LIGHT_KMS
                pred = C_LIGHT_KMS / dh_over_rd  # This gives H(z)*r_d
            else:
                pred = getattr(model, method_name)(dp.z_eff, params, rd)

            if is_inverse and dp.obs_name not in ("DA_over_rs", "Hz_rs"):
                pred = 1.0 / pred

            theory[i] = pred
        return theory

    def log_likelihood(
        self,
        model: CosmologicalModel,
        params: dict[str, float],
        rd: float = DEFAULT_RD,
    ) -> float:
        """Compute the Gaussian log-likelihood.

        Returns -0.5 * delta^T C^{-1} delta  (unnormalized).
        """
        theory = self.theory_vector(model, params, rd)
        delta = self.data_vector - theory
        return -0.5 * delta @ self.inv_cov @ delta
