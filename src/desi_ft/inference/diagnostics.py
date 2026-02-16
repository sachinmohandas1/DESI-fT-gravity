"""MCMC convergence diagnostics and trace plots."""

from __future__ import annotations

from pathlib import Path

import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np


def autocorrelation_summary(sampler: emcee.EnsembleSampler, param_names: list[str]) -> np.ndarray:
    """Compute and print integrated autocorrelation times.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Sampler after running MCMC.
    param_names : list[str]
        Names of the free parameters.

    Returns
    -------
    np.ndarray
        Autocorrelation times for each parameter.
    """
    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except emcee.autocorr.AutocorrError:
        print("WARNING: Chain may be too short for reliable autocorrelation estimate.")
        tau = sampler.get_autocorr_time(quiet=True, tol=0)

    print("\nAutocorrelation times:")
    for name, t in zip(param_names, tau):
        print(f"  {name}: {t:.1f}")
    print(f"  Max: {np.max(tau):.1f}")

    n_steps = sampler.iteration
    print(f"  Chain length / max(tau): {n_steps / np.max(tau):.1f} (want > 50)")
    return tau


def gelman_rubin(sampler: emcee.EnsembleSampler, n_burn: int = 0) -> np.ndarray:
    """Compute the Gelman-Rubin R-hat statistic.

    Treats each walker as an independent chain. For emcee this is
    an approximation since walkers are correlated, but it's still
    a useful diagnostic.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Sampler after running MCMC.
    n_burn : int
        Number of burn-in steps to discard.

    Returns
    -------
    np.ndarray
        R-hat for each parameter.
    """
    # Shape: (n_walkers, n_steps, n_dim)
    chain = sampler.get_chain(discard=n_burn)  # (n_steps, n_walkers, n_dim)
    chain = chain.transpose(1, 0, 2)  # (n_walkers, n_steps, n_dim)
    n_chains, n_steps, n_dim = chain.shape

    # Within-chain variance
    W = np.mean(np.var(chain, axis=1, ddof=1), axis=0)

    # Between-chain variance
    chain_means = np.mean(chain, axis=1)  # (n_chains, n_dim)
    overall_mean = np.mean(chain_means, axis=0)  # (n_dim,)
    B = n_steps / (n_chains - 1) * np.sum(
        (chain_means - overall_mean) ** 2, axis=0
    )

    # Pooled variance estimate
    V_hat = (1.0 - 1.0 / n_steps) * W + B / n_steps
    R_hat = np.sqrt(V_hat / W)
    return R_hat


def trace_plot(
    sampler: emcee.EnsembleSampler,
    param_names: list[str],
    output_path: str | Path | None = None,
    n_burn: int = 0,
) -> plt.Figure:
    """Generate trace plots for each parameter.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Sampler after running MCMC.
    param_names : list[str]
        Names of the free parameters.
    output_path : str or Path, optional
        If given, save the figure.
    n_burn : int
        If > 0, draw a vertical line at the burn-in cutoff.

    Returns
    -------
    matplotlib.figure.Figure
    """
    chain = sampler.get_chain()  # (n_steps, n_walkers, n_dim)
    n_dim = chain.shape[2]

    fig, axes = plt.subplots(n_dim, 1, figsize=(10, 3 * n_dim), sharex=True)
    if n_dim == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], alpha=0.3, lw=0.5)
        ax.set_ylabel(param_names[i])
        if n_burn > 0:
            ax.axvline(n_burn, color="r", ls="--", lw=1, label="burn-in")
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Step")
    fig.suptitle("MCMC Trace Plots", y=1.01)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Trace plot saved to {output_path}")

    return fig
