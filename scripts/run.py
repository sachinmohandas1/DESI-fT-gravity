#!/usr/bin/env python
"""Run an MCMC analysis from a YAML configuration file.

Usage::

    python scripts/run.py configs/lcdm_desi_bao.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import yaml

from desi_ft.models.lcdm import LCDM
from desi_ft.likelihoods.bao import BAOLikelihood
from desi_ft.inference.mcmc import run_mcmc
from desi_ft.inference.diagnostics import autocorrelation_summary, gelman_rubin, trace_plot
from desi_ft.plotting.contours import triangle_plot

_MODEL_REGISTRY: dict[str, type] = {
    "lcdm": LCDM,
}


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Instantiate model
    model_name = config["model"]["name"]
    if model_name not in _MODEL_REGISTRY:
        print(f"Unknown model: {model_name}. Available: {list(_MODEL_REGISTRY)}")
        sys.exit(1)
    model = _MODEL_REGISTRY[model_name]()

    # Instantiate likelihood
    bao_cfg = config["likelihood"]["bao"]
    likelihood = BAOLikelihood(bao_cfg["data_file"], bao_cfg["cov_file"])
    print(f"Loaded {len(likelihood.data_points)} BAO data points")

    # Run MCMC
    sampler = run_mcmc(model, likelihood, config)

    # Diagnostics
    n_burn = config["sampler"].get("n_burn", 1000)

    # Read free param names from chain metadata
    import h5py
    chain_dir = Path(config["output"]["chain_dir"])
    chain_file = chain_dir / "chain.h5"
    with h5py.File(str(chain_file), "r") as hf:
        free_names = list(hf["metadata"].attrs["free_param_names"])

    tau = autocorrelation_summary(sampler, free_names)
    r_hat = gelman_rubin(sampler, n_burn=n_burn)
    print(f"\nGelman-Rubin R-hat: {dict(zip(free_names, r_hat))}")
    for name, rh in zip(free_names, r_hat):
        if rh > 1.1:
            print(f"  WARNING: R-hat for {name} = {rh:.3f} > 1.1 — chain may not be converged")

    # Trace plot
    fig_dir = Path(config["output"]["figure_dir"])
    trace_plot(sampler, free_names, output_path=fig_dir / "trace.png", n_burn=n_burn)

    # Triangle plot
    triangle_plot(chain_file, output_path=fig_dir / "triangle.png")

    # Print summary statistics
    flat_samples = sampler.get_chain(discard=n_burn, flat=True)
    print("\n=== Posterior Summary ===")
    for i, name in enumerate(free_names):
        q = [16, 50, 84]
        lo, med, hi = [float(x) for x in __import__("numpy").percentile(flat_samples[:, i], q)]
        print(f"  {name} = {med:.4f}  (+{hi - med:.4f} / -{med - lo:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DESI f(T) gravity MCMC")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
