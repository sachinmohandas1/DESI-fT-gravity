"""Tests for cosmological distance calculations.

Validation targets: LCDM with Omega_m=0.3, H0=70 km/s/Mpc, standard r_d.
"""

import numpy as np
import pytest

from desi_ft.cosmology.distances import (
    C_LIGHT_KMS,
    comoving_distance,
    hubble_distance,
    volume_averaged_distance,
)
from desi_ft.models.lcdm import LCDM


@pytest.fixture
def lcdm_H():
    """Return H(z) callable for a reference LCDM cosmology."""
    model = LCDM()
    params = {"Omega_m": 0.3, "H0": 70.0}
    return lambda z: model.hubble(z, params)


def test_hubble_distance_z0(lcdm_H):
    """D_H(z=0) = c / H0."""
    dh = hubble_distance(0.0, lcdm_H)
    assert dh == pytest.approx(C_LIGHT_KMS / 70.0, rel=1e-10)


def test_hubble_distance_z1(lcdm_H):
    """D_H(z=1) = c / H(1).  H(1) > H0 so D_H(1) < D_H(0)."""
    dh0 = hubble_distance(0.0, lcdm_H)
    dh1 = hubble_distance(1.0, lcdm_H)
    assert dh1 < dh0


def test_comoving_distance_z0(lcdm_H):
    """D_M(z=0) = 0."""
    dm = comoving_distance(0.0, lcdm_H)
    assert dm == pytest.approx(0.0, abs=1e-10)


def test_comoving_distance_z1(lcdm_H):
    """D_M(z=1) for reference LCDM should be ~3306 Mpc.

    Using Omega_m=0.3, H0=70, Omega_r~0, Omega_Lambda~0.7.
    Astropy gives D_M(1) ≈ 3306 Mpc for these params.
    We allow 1% tolerance to account for our small Omega_r.
    """
    dm = comoving_distance(1.0, lcdm_H)
    assert dm == pytest.approx(3306.0, rel=0.01)


def test_comoving_distance_monotonic(lcdm_H):
    """D_M should be monotonically increasing with z."""
    zs = [0.1, 0.5, 1.0, 2.0, 3.0]
    dms = [comoving_distance(z, lcdm_H) for z in zs]
    for i in range(len(dms) - 1):
        assert dms[i] < dms[i + 1]


def test_volume_averaged_distance_relation(lcdm_H):
    """D_V = (z * D_M^2 * D_H)^{1/3}."""
    z = 0.5
    dm = comoving_distance(z, lcdm_H)
    dh = hubble_distance(z, lcdm_H)
    dv = volume_averaged_distance(z, lcdm_H)
    expected = (z * dm**2 * dh) ** (1.0 / 3.0)
    assert dv == pytest.approx(expected, rel=1e-10)


def test_array_input(lcdm_H):
    """Distance functions should accept arrays."""
    zs = np.array([0.5, 1.0, 1.5])
    dm = comoving_distance(zs, lcdm_H)
    assert dm.shape == (3,)
    dh = hubble_distance(zs, lcdm_H)
    assert dh.shape == (3,)
    dv = volume_averaged_distance(zs, lcdm_H)
    assert dv.shape == (3,)
