"""Flat LCDM cosmological model.

This is f(T) = T - 2*Lambda, which reproduces the standard Friedmann
equation.  It serves as the baseline for comparison with modified
gravity models in later phases.
"""

from __future__ import annotations

import numpy as np

from .base import CosmologicalModel

# Effective number of neutrino species (standard value)
N_EFF = 3.044


class LCDM(CosmologicalModel):
    """Flat LCDM with parameters (Omega_m, H0).

    The radiation density is computed self-consistently from the CMB
    temperature T_CMB = 2.7255 K via

        Omega_r = 4.15e-5 / h^2 * (1 + 0.2271 * N_eff)

    and flatness fixes Omega_Lambda = 1 - Omega_m - Omega_r.
    """

    @property
    def param_names(self) -> list[str]:
        return ["Omega_m", "H0"]

    @property
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "Omega_m": (0.1, 0.9),
            "H0": (50.0, 100.0),
        }

    @staticmethod
    def _omega_r(H0: float) -> float:
        """Radiation density parameter today including neutrinos."""
        h = H0 / 100.0
        return 4.15e-5 / h**2 * (1.0 + 0.2271 * N_EFF)

    def hubble(self, z: float, params: dict[str, float]) -> float:
        """H(z) = H0 * sqrt(Omega_m*(1+z)^3 + Omega_r*(1+z)^4 + Omega_Lambda)."""
        Om = params["Omega_m"]
        H0 = params["H0"]
        Or = self._omega_r(H0)
        OL = 1.0 - Om - Or
        zp1 = 1.0 + z
        return H0 * np.sqrt(Om * zp1**3 + Or * zp1**4 + OL)
