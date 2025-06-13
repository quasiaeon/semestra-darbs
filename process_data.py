#!/usr/bin/env python
"""
process_data.py – clean Gaia DR3 parallaxes and estimate distances.

Windows-safe version: no healpy, no local dustmaps; extinction comes from the
Bayestar-2019 web API (optional, slower).
"""
from __future__ import annotations
import argparse, io, json, math, requests
from pathlib import Path
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# --------------------------------------------------------------------------- #
# Parallax zero-point (Lindegren 2021, Eq. B.1)
# --------------------------------------------------------------------------- #
def parallax_zp(g, bp_rp, beta):
    C = np.array([
        -0.017, +0.003, +0.007,
        +0.024, -0.023, -0.004,
        -0.012, +0.007, +0.004,
        -0.005, +0.000, +0.000,
        +0.001, +0.000, +0.000
    ])
    x, y = g - 12.0, bp_rp - 1.5
    z = np.sin(np.deg2rad(beta))
    terms = np.column_stack([
        np.ones_like(x), x, y,
        z, x*y, x*z,
        y*z, x**2, y**2,
        z**2, x**2*y, x*y**2,
        x**2*z, x*z**2, y**2*z
    ])
    return (C @ terms.T).ravel()          # [mas]

# --------------------------------------------------------------------------- #
# Bailer-Jones distance mode (vectorised)
# --------------------------------------------------------------------------- #
def distance_bailer_jones(pi, sig, L=1.35e3):
    inv = 1_000.0 / pi
    good = (sig / pi) < 0.2
    r = np.full_like(pi, np.nan)
    r[good] = inv[good]
    w = ~good
    var = sig[w]**2
    term = 1 - 4 * L * pi[w] * var
    term[term < 0] = np.nan
    r[w] = 0.5 / pi[w] * (1 + np.sqrt(term))
    return r

# --------------------------------------------------------------------------- #
# Extinction from Bayestar 2019 web API (no healpy)
# --------------------------------------------------------------------------- #
def estimate_Av_web(df: pd.DataFrame, chunk=100):
    url = "https://argonaut.skymaps.info/gal-l2?version=2019"
    Av = np.zeros(len(df))
    for i in range(0, len(df), chunk):
        sub = df.iloc[i:i+chunk]
        c = SkyCoord(sub.ra.values*u.deg,
                     sub.dec.values*u.deg,
                     distance=sub.distance_pc.values*u.pc,
                     frame="icrs").galactic
        buf = io.StringIO()
        buf.write("l,b,d\n")
        for l, b, d in zip(c.l.deg, c.b.deg, sub.distance_pc.values):
            buf.write(f"{l:.6f},{b:.6f},{d:.1f}\n")
        try:
            r = requests.post(url,
                              files={"file": ("coords.csv", buf.getvalue())},
                              timeout=60)
            r.raise_for_status()
            Av[i:i+chunk] = [rec["A0"] for rec in
                             map(json.loads, r.text.strip().splitlines())]
        except Exception as exc:
            print(f"⚠ web-extinction chunk failed ({exc}); Av=0")
    return Av

# --------------------------------------------------------------------------- #
def clean_and_compute(df: pd.DataFrame, extinction: str) -> pd.DataFrame:
    beta = df.get("ecl_lat", pd.Series(np.zeros(len(df)))).values
    zp = parallax_zp(df.phot_g_mean_mag.values,
                     df.bp_rp.values, beta)
    pi_corr = df.parallax.values - zp
    sig = df.parallax_error.values

    # quality cuts (skip if column absent)
    ruwe_ok = df.get("ruwe", pd.Series(np.ones(len(df))*1.0)) < 1.4
    excess_ok = df.get("phot_bp_rp_excess_factor",
                       pd.Series(np.ones(len(df))*1.2)).between(1.0, 1.8)
        # keep only stars that pass quality + positive parallax
    keep = ruwe_ok & excess_ok & (pi_corr > 0)
    df   = df.loc[keep].reset_index(drop=True)
    if df.empty:
        return df.assign(distance_pc=[], distance_error_pc=[], Av=[])

    # ---- distances (use the filtered arrays!) ----------------------------
    pi_use  = pi_corr[keep]
    sig_use = sig[keep]
    dist     = distance_bailer_jones(pi_use, sig_use)
    dist_err = dist * (sig_use / pi_use)
    df = df.assign(distance_pc=dist, distance_error_pc=dist_err)

    # extinction
    if extinction == "web":
        Av = estimate_Av_web(df)
    else:                                   # "none"
        Av = np.zeros(len(df))
    return df.assign(Av=Av)

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  default="data/stars_raw.parquet")
    ap.add_argument("--out", dest="out",  default="data/stars_clean.parquet")
    ap.add_argument("--ext", choices=["none", "web"], default="none",
                    help="Extinction mode: none (fast) or web (Bayestar API)")
    args = ap.parse_args()

    df_raw = pd.read_parquet(args.inp)
    df_clean = clean_and_compute(df_raw, args.ext)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(args.out, index=False)

    ok = df_clean.distance_pc.notna().sum()
    print(f"✓ kept {len(df_clean):,}/{len(df_raw):,} stars ({ok:,} with distances)")

if __name__ == "__main__":
    main()