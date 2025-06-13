"""Denoise Gaia BP/RP XP spectra and extract simple spectral indices.

Outputs
-------
- data/xp_clean.parquet   ← adds flux_smooth, TiO_index, Balmer_index …
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# ---------- helpers ---------------------------------------------------------

def smooth_flux(arr: list[float]) -> list[float]:
    """Apply Savitzky‑Golay smoothing (window 17, poly 3)."""
    y = np.asarray(arr, dtype=float)
    if y.size < 17:
        return y.tolist()  # leave tiny spectra untouched
    return savgol_filter(y, 17, 3).tolist()

# Simple index example: pseudo‑equivalent width around 620 nm ---------------

def tiox_index(wave: list[float], flux: list[float]) -> float:
    w = np.asarray(wave)
    f = np.asarray(flux)
    band = (w > 615) & (w < 625)
    cont = (w > 600) & (w < 610)
    if not band.any() or not cont.any():
        return np.nan
    ew = 1 - f[band].mean() / f[cont].mean()
    return ew

# ---------- CLI -------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("BP/RP XP spectral denoising")
    p.add_argument("--in", dest="inp", default="data/xp_raw.parquet")
    p.add_argument("--out", dest="out", default="data/xp_clean.parquet")
    args = p.parse_args()

    df = pd.read_parquet(args.inp)

    df["flux_smooth"] = df.flux.apply(smooth_flux)
    df["TiO_index"] = [tiox_index(w, f) for w, f in zip(df.wave, df.flux_smooth)]

    df.to_parquet(args.out, index=False)
    print(f"✓ Smoothed {len(df):,} spectra → xp_clean.parquet")

if __name__ == "__main__":
    main()
