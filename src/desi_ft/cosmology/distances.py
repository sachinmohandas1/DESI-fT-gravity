"""Cosmological distance measures via numerical integration."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad

# Speed of light in km/s
C_LIGHT_KMS = 299792.458


def comoving_distance(
    z: float | np.ndarray,
    H_func: Callable[[float], float],
) -> float | np.ndarray:
    """Compute the comoving (transverse) distance D_M(z) for flat geometry.

    D_M(z) = c * integral_0^z dz' / H(z')

    Parameters
    ----------
    z : float or array
        Redshift(s).
    H_func : callable
        H(z) in km/s/Mpc.

    Returns
    -------
    float or array
        D_M in Mpc.
    """
    scalar = np.isscalar(z)
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))

    result = np.empty_like(z_arr)
    for i, zi in enumerate(z_arr):
        val, _ = quad(lambda zp: C_LIGHT_KMS / H_func(zp), 0.0, zi)
        result[i] = val

    return float(result[0]) if scalar else result


def hubble_distance(
    z: float | np.ndarray,
    H_func: Callable[[float], float],
) -> float | np.ndarray:
    """Compute the Hubble distance D_H(z) = c / H(z).

    Parameters
    ----------
    z : float or array
        Redshift(s).
    H_func : callable
        H(z) in km/s/Mpc.

    Returns
    -------
    float or array
        D_H in Mpc.
    """
    scalar = np.isscalar(z)
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    result = C_LIGHT_KMS / np.array([H_func(float(zi)) for zi in z_arr])
    return float(result[0]) if scalar else result


def volume_averaged_distance(
    z: float | np.ndarray,
    H_func: Callable[[float], float],
) -> float | np.ndarray:
    """Compute the volume-averaged distance D_V(z).

    D_V(z) = [z * D_M(z)^2 * D_H(z)]^{1/3}

    Parameters
    ----------
    z : float or array
        Redshift(s).
    H_func : callable
        H(z) in km/s/Mpc.

    Returns
    -------
    float or array
        D_V in Mpc.
    """
    dm = comoving_distance(z, H_func)
    dh = hubble_distance(z, H_func)
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    dm_arr = np.atleast_1d(dm)
    dh_arr = np.atleast_1d(dh)

    result = (z_arr * dm_arr**2 * dh_arr) ** (1.0 / 3.0)

    if np.isscalar(z):
        return float(result[0])
    return result
