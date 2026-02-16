# DESI f(T) Gravity

Bayesian inference pipeline for constraining **f(T) teleparallel gravity** models using DESI baryon acoustic oscillation (BAO) data.

## Overview

The [DESI collaboration](https://www.desi.lbl.gov/) has measured BAO distances in 7 redshift bins from millions of extragalactic objects, finding a 2.5–3.9σ preference for evolving dark energy over ΛCDM.

This project tests whether **f(T) teleparallel gravity** — modified gravity theories based on torsion rather than curvature — can explain the DESI signal. We implement several f(T) models, compute BAO observable predictions, and constrain parameters via MCMC sampling.

### Models

| Model | f(T) | Extra params | Reference |
|-------|------|-------------|-----------|
| **ΛCDM** (baseline) | T − 2Λ | 0 | — |
| Power-law | α(−T)^b | 1 (b) | Bengochea & Ferraro 2009 |
| Exponential IR | T·exp(β T₀/T) | 1 (β) | Linder 2010 |
| Sqrt-exponential | T + α√(−T)·exp(−γ/T) | 2 | — |

## Installation

```bash
git clone https://github.com/sachinmohandas1/DESI-fT-gravity.git
cd DESI-fT-gravity
pip install -e ".[dev]"
```

## Quickstart

### 1. Download DESI BAO data

```bash
python scripts/download_data.py
```

This downloads the public DESI DR1 BAO likelihood files from the [CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data) repository.

### 2. Run ΛCDM validation

```bash
python scripts/run.py configs/lcdm_desi_bao.yaml
```

This runs an emcee MCMC sampling the ΛCDM model against DESI BAO data. The expected result is **Ωm ≈ 0.295 ± 0.015** (consistent with [arXiv:2404.03002](https://arxiv.org/abs/2404.03002)).

Chains are saved to `chains/` and figures to `figures/`.

### 3. Run tests

```bash
pytest
```

## Project Structure

```
src/desi_ft/
├── cosmology/distances.py    # D_M, D_H, D_V numerical integration
├── models/
│   ├── base.py               # Abstract CosmologicalModel base class
│   └── lcdm.py               # Flat ΛCDM baseline
├── likelihoods/bao.py        # Gaussian BAO likelihood (cobaya format)
├── inference/
│   ├── mcmc.py               # emcee ensemble sampler wrapper
│   └── diagnostics.py        # Autocorrelation, R-hat, trace plots
└── plotting/contours.py      # GetDist triangle plots
```

## Roadmap

- **Phase 1** (current): ΛCDM validation against DESI DR1 BAO
- **Phase 2**: Implement power-law, exponential, sqrt-exponential f(T) models
- **Phase 3**: Add Pantheon+ SN Ia, cosmic chronometers, Planck CMB priors
- **Phase 4**: Nested sampling (dynesty) for Bayesian model comparison
- **Phase 5**: Perturbation-theory analysis and growth-rate constraints

## References

- DESI DR1 BAO: [arXiv:2404.03002](https://arxiv.org/abs/2404.03002)
- Bengochea & Ferraro (2009): [arXiv:0812.1205](https://arxiv.org/abs/0812.1205)
- Linder (2010): [arXiv:1005.3039](https://arxiv.org/abs/1005.3039)

## License

MIT
