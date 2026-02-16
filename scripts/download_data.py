#!/usr/bin/env python
"""Download DESI DR1 BAO data files from the CobayaSampler/bao_data repository.

Usage::

    python scripts/download_data.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master"
OUTPUT_DIR = Path("data/desi_dr1_bao")

# DR1 files (April 2024)
DR1_FILES = [
    # Combined (all tracers)
    "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",
    "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt",
    # BGS (z = 0.1-0.4)
    "desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_mean.txt",
    "desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_cov.txt",
    # LRG low-z (z = 0.4-0.6)
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt",
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_cov.txt",
    # LRG high-z (z = 0.6-0.8)
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt",
    "desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_cov.txt",
    # LRG+ELG (z = 0.8-1.1)
    "desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_mean.txt",
    "desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_cov.txt",
    # ELG (z = 1.1-1.6)
    "desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt",
    "desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_cov.txt",
    # QSO (z = 0.8-2.1)
    "desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_mean.txt",
    "desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_cov.txt",
    # Lya (z ~ 2.33)
    "desi_2024_gaussian_bao_Lya_GCcomb_mean.txt",
    "desi_2024_gaussian_bao_Lya_GCcomb_cov.txt",
    # eBOSS+DESI combined Lya
    "desi_2024_eboss_gaussian_bao_Lya_GCcomb_mean.txt",
    "desi_2024_eboss_gaussian_bao_Lya_GCcomb_cov.txt",
]


def download() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in DR1_FILES:
        url = f"{BASE_URL}/{filename}"
        dest = OUTPUT_DIR / filename
        if dest.exists():
            print(f"  [skip] {filename} (already exists)")
            continue
        print(f"  [download] {filename}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nDone. Files saved to {OUTPUT_DIR}/")
    print(f"Total: {sum(1 for f in OUTPUT_DIR.iterdir() if f.is_file())} files")


if __name__ == "__main__":
    download()
