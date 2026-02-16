"""Abstract base class for cosmological models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from desi_ft.cosmology.distances import (
    comoving_distance,
    hubble_distance,
    volume_averaged_distance,
)

C_LIGHT_KMS = 299792.458
DEFAULT_RD = 147.09  # Sound horizon at drag epoch in Mpc (Planck 2018)


class CosmologicalModel(ABC):
    """Base class that all cosmological models must subclass.

    Subclasses implement ``hubble(z, params)``; distance ratios are
    computed automatically from the Hubble function.
    """

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Ordered list of free parameter names."""

    @property
    @abstractmethod
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        """Prior bounds for each parameter: {name: (lo, hi)}."""

    @abstractmethod
    def hubble(self, z: float, params: dict[str, float]) -> float:
        """Return H(z) in km/s/Mpc for the given parameter values."""

    # ------------------------------------------------------------------
    # Derived distance ratios (concrete; call hubble internally)
    # ------------------------------------------------------------------

    def _H_func(self, params: dict[str, float]):
        """Return a closure H(z) suitable for the distance integrators."""
        return lambda z: self.hubble(z, params)

    def DM_over_rd(
        self, z: float | np.ndarray, params: dict[str, float], rd: float = DEFAULT_RD
    ) -> float | np.ndarray:
        """Transverse comoving distance divided by sound horizon."""
        return comoving_distance(z, self._H_func(params)) / rd

    def DH_over_rd(
        self, z: float | np.ndarray, params: dict[str, float], rd: float = DEFAULT_RD
    ) -> float | np.ndarray:
        """Hubble distance divided by sound horizon."""
        return hubble_distance(z, self._H_func(params)) / rd

    def DV_over_rd(
        self, z: float | np.ndarray, params: dict[str, float], rd: float = DEFAULT_RD
    ) -> float | np.ndarray:
        """Volume-averaged distance divided by sound horizon."""
        return volume_averaged_distance(z, self._H_func(params)) / rd

    # ------------------------------------------------------------------
    # Prior check
    # ------------------------------------------------------------------

    def in_prior(self, params: dict[str, float]) -> bool:
        """Return True if all parameters are within their prior bounds."""
        for name, (lo, hi) in self.param_bounds.items():
            val = params.get(name)
            if val is None or not (lo <= val <= hi):
                return False
        return True
