"""Ensemble MCMC sampling with emcee."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import emcee
import h5py
import numpy as np

from desi_ft.models.base import CosmologicalModel, DEFAULT_RD
from desi_ft.likelihoods.bao import BAOLikelihood


def _log_posterior(
    theta: np.ndarray,
    model: CosmologicalModel,
    likelihood: BAOLikelihood,
    param_names: list[str],
    rd: float,
) -> float:
    """Log-posterior = flat prior + Gaussian likelihood."""
    params = dict(zip(param_names, theta))

    # Flat prior check
    if not model.in_prior(params):
        return -np.inf

    return likelihood.log_likelihood(model, params, rd)


def run_mcmc(
    model: CosmologicalModel,
    likelihood: BAOLikelihood,
    config: dict[str, Any],
) -> emcee.EnsembleSampler:
    """Run emcee ensemble MCMC and save chains to HDF5.

    Parameters
    ----------
    model : CosmologicalModel
        The cosmological model to sample.
    likelihood : BAOLikelihood
        The BAO likelihood object.
    config : dict
        Must contain keys from the YAML config:
        - sampler.n_walkers (default 32)
        - sampler.n_steps (default 5000)
        - sampler.n_burn (default 1000)
        - model.params.<name>.start for each parameter
        - model.params.<name>.fixed (optional)
        - output.chain_dir

    Returns
    -------
    emcee.EnsembleSampler
    """
    sampler_cfg = config.get("sampler", {})
    n_walkers = sampler_cfg.get("n_walkers", 32)
    n_steps = sampler_cfg.get("n_steps", 5000)
    n_burn = sampler_cfg.get("n_burn", 1000)

    model_cfg = config.get("model", {}).get("params", {})

    # Identify free vs fixed parameters
    param_names = model.param_names
    free_names: list[str] = []
    fixed_params: dict[str, float] = {}
    starts: list[float] = []

    for name in param_names:
        pcfg = model_cfg.get(name, {})
        if isinstance(pcfg, dict) and "fixed" in pcfg:
            fixed_params[name] = pcfg["fixed"]
        else:
            free_names.append(name)
            start = pcfg.get("start", None) if isinstance(pcfg, dict) else None
            if start is None:
                lo, hi = model.param_bounds[name]
                start = 0.5 * (lo + hi)
            starts.append(start)

    # Get r_d (may be fixed or free)
    rd = model_cfg.get("r_d", {})
    if isinstance(rd, dict):
        rd = rd.get("fixed", DEFAULT_RD)
    else:
        rd = float(rd) if rd else DEFAULT_RD

    ndim = len(free_names)
    starts_arr = np.array(starts)

    # Initialize walkers as small Gaussian ball around starting point
    rng = np.random.default_rng(42)
    pos = starts_arr + 1e-3 * rng.standard_normal((n_walkers, ndim))

    # Build the full-params closure
    def log_prob(theta: np.ndarray) -> float:
        params = dict(fixed_params)
        for i, name in enumerate(free_names):
            params[name] = theta[i]
        return _log_posterior(
            np.array([params[n] for n in param_names]),
            model, likelihood, param_names, rd,
        )

    # Set up HDF5 backend
    chain_dir = Path(config.get("output", {}).get("chain_dir", "chains/default"))
    chain_dir.mkdir(parents=True, exist_ok=True)
    backend_path = chain_dir / "chain.h5"
    backend = emcee.backends.HDFBackend(str(backend_path))
    backend.reset(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_prob, backend=backend,
    )

    print(f"Running MCMC: {n_walkers} walkers, {n_steps} steps, {ndim} free params")
    print(f"Free parameters: {free_names}")
    print(f"Fixed parameters: {fixed_params}")
    print(f"Saving chains to: {backend_path}")

    sampler.run_mcmc(pos, n_steps, progress=True)

    # Store metadata
    with h5py.File(str(backend_path), "a") as f:
        meta = f.require_group("metadata")
        meta.attrs["free_param_names"] = free_names
        meta.attrs["n_burn"] = n_burn
        meta.attrs["rd"] = rd
        for k, v in fixed_params.items():
            meta.attrs[f"fixed_{k}"] = v

    print(f"Done. Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    return sampler
