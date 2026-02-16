"""Publication-quality contour (triangle) plots via GetDist."""

from __future__ import annotations

from pathlib import Path

import emcee
import h5py
import numpy as np
from getdist import MCSamples, plots


def triangle_plot(
    chain_file: str | Path,
    param_names: list[str] | None = None,
    labels: list[str] | None = None,
    output_path: str | Path | None = None,
    n_burn: int | None = None,
) -> plots.GetDistPlotter:
    """Create a triangle (corner) plot from an emcee HDF5 chain file.

    Parameters
    ----------
    chain_file : str or Path
        Path to the emcee HDF5 backend file.
    param_names : list[str], optional
        Parameter names. If None, read from file metadata.
    labels : list[str], optional
        LaTeX labels for parameters. If None, uses param_names.
    output_path : str or Path, optional
        If given, save the figure.
    n_burn : int, optional
        Burn-in steps to discard. If None, read from file metadata.

    Returns
    -------
    getdist.plots.GetDistPlotter
    """
    chain_file = Path(chain_file)

    # Load chain from HDF5
    reader = emcee.backends.HDFBackend(str(chain_file), read_only=True)

    with h5py.File(str(chain_file), "r") as f:
        meta = f.get("metadata", {})
        if param_names is None:
            param_names = list(meta.attrs.get("free_param_names", []))
        if n_burn is None:
            n_burn = int(meta.attrs.get("n_burn", 0))

    # Flatten chain: discard burn-in, thin by 1 (can add thinning later)
    samples = reader.get_chain(discard=n_burn, flat=True)  # (n_samples, n_dim)
    log_prob = reader.get_log_prob(discard=n_burn, flat=True)

    if labels is None:
        _default_labels = {
            "Omega_m": r"\Omega_m",
            "H0": r"H_0",
            "r_d": r"r_d",
        }
        labels = [_default_labels.get(n, n) for n in param_names]

    # Build GetDist MCSamples
    mc_samples = MCSamples(
        samples=samples,
        loglikes=-log_prob,  # GetDist uses -log(L) convention
        names=param_names,
        labels=labels,
        name_tag="DESI BAO",
    )

    # Triangle plot
    g = plots.get_subplot_plotter()
    g.triangle_plot(mc_samples, filled=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        g.export(str(output_path))
        print(f"Triangle plot saved to {output_path}")

    return g
