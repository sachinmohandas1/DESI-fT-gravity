"""Tests for the LCDM model."""

import numpy as np
import pytest

from desi_ft.models.lcdm import LCDM


@pytest.fixture
def model():
    return LCDM()


@pytest.fixture
def fiducial_params():
    return {"Omega_m": 0.3, "H0": 70.0}


def test_hubble_z0(model, fiducial_params):
    """H(z=0) should equal H0."""
    H0 = fiducial_params["H0"]
    assert model.hubble(0.0, fiducial_params) == pytest.approx(H0, rel=1e-10)


def test_hubble_monotonic(model, fiducial_params):
    """H(z) must be monotonically increasing for LCDM (matter + Lambda dominated)."""
    zs = np.linspace(0, 5, 100)
    Hs = [model.hubble(z, fiducial_params) for z in zs]
    for i in range(len(Hs) - 1):
        assert Hs[i] <= Hs[i + 1]


def test_omega_lambda_positive(model):
    """Omega_Lambda = 1 - Omega_m - Omega_r should be positive for physical params."""
    H0 = 70.0
    Omega_r = model._omega_r(H0)
    Omega_m = 0.3
    Omega_Lambda = 1.0 - Omega_m - Omega_r
    assert Omega_Lambda > 0
    assert Omega_Lambda == pytest.approx(0.7, abs=0.01)


def test_omega_r_small(model):
    """Omega_r should be ~9e-5, much smaller than Omega_m."""
    Omega_r = model._omega_r(70.0)
    assert 5e-5 < Omega_r < 2e-4


def test_param_names(model):
    assert model.param_names == ["Omega_m", "H0"]


def test_param_bounds(model):
    bounds = model.param_bounds
    assert "Omega_m" in bounds
    assert "H0" in bounds
    assert bounds["Omega_m"] == (0.1, 0.9)
    assert bounds["H0"] == (50.0, 100.0)


def test_in_prior(model):
    assert model.in_prior({"Omega_m": 0.3, "H0": 70.0}) is True
    assert model.in_prior({"Omega_m": 0.05, "H0": 70.0}) is False
    assert model.in_prior({"Omega_m": 0.3, "H0": 120.0}) is False


def test_distance_ratios(model, fiducial_params):
    """Smoke test: distance ratios should be positive and finite."""
    z = 0.5
    rd = 147.09
    dm_rd = model.DM_over_rd(z, fiducial_params, rd)
    dh_rd = model.DH_over_rd(z, fiducial_params, rd)
    dv_rd = model.DV_over_rd(z, fiducial_params, rd)

    assert np.isfinite(dm_rd) and dm_rd > 0
    assert np.isfinite(dh_rd) and dh_rd > 0
    assert np.isfinite(dv_rd) and dv_rd > 0


def test_high_redshift_radiation(model):
    """At high z, radiation term should matter — H(z=1000) >> H(z=0)."""
    params = {"Omega_m": 0.3, "H0": 70.0}
    H_high = model.hubble(1000.0, params)
    H_low = model.hubble(0.0, params)
    assert H_high / H_low > 1e4
