#!/usr/bin/env python
"""
error_analysis.py  –  Propagate parallax errors to voxel σ(N)
-------------------------------------------------------------
* Run with no arguments and it will auto-locate:
    data/stars_clean.parquet
    output/voxel_cube.npz
* Or override paths with --stars and/or --cube.

Produces
--------
output/error_cube.npz
    sigma, R_edges, phi_edges, Z_edges
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.random import default_rng
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u


# --------------------------------------------------------------------
# Default helpers
# --------------------------------------------------------------------
PROJ_ROOT = Path.cwd()
DEF_STARS = PROJ_ROOT / "data"   / "stars_clean.parquet"
DEF_CUBE  = PROJ_ROOT / "output" / "voxel_cube.npz"
DEF_OUT   = PROJ_ROOT / "output"

def auto_path(pth: Path, label: str) -> Path:
    if pth.exists():
        print(f"[auto] using {pth} as {label}")
        return pth
    sys.exit(f"ERROR: cannot auto-detect {label}. "
             "Run with explicit --stars and --cube.")


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser("Monte-Carlo voxel uncertainty")
    p.add_argument("--stars", help="Parquet with cleaned star catalogue")
    p.add_argument("--cube",  help="voxel_cube.npz with bin edges")
    p.add_argument("--draws", type=int, default=1000,
                   help="Monte-Carlo realisations [1000]")
    p.add_argument("--outdir", default=None,
                   help=f"Destination (default: {DEF_OUT})")
    p.add_argument("--seed",   type=int, default=42,
                   help="Random seed [42]")
    return p.parse_args()


# --------------------------------------------------------------------
# Cylindrical coords
# --------------------------------------------------------------------
def cylindrical(df: pd.DataFrame) -> np.ndarray:
    R   = np.hypot(df.x.values, df.y.values)
    phi = np.degrees(np.arctan2(df.y.values, df.x.values))
    return np.column_stack([R, phi, df.z.values])

# --------------------------------------------------------------------
# If x,y,z are missing, derive them from (ra, dec, distance_pc)
# --------------------------------------------------------------------
def ensure_galactocentric(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with x,y,z [kpc] columns added if absent."""
    if {"x", "y", "z"}.issubset(df.columns):
        return df

    print("→ Deriving Galactocentric (x,y,z) via Astropy …")
    c_icrs = SkyCoord(ra=df.ra.values * u.deg,
                      dec=df.dec.values * u.deg,
                      distance=df.distance_pc.values * u.pc,
                      frame="icrs")
    gal = c_icrs.transform_to(Galactocentric)
    df["x"] = gal.x.to_value(u.kpc)
    df["y"] = gal.y.to_value(u.kpc)
    df["z"] = gal.z.to_value(u.kpc)
    return df

# --------------------------------------------------------------------
# MC propagation
# --------------------------------------------------------------------
def mc_errors(df: pd.DataFrame, edges: tuple[np.ndarray, ...],
              draws: int, rng) -> np.ndarray:
    """Streaming σ(N) via Welford; RAM << full (draw, …) stack."""
    R_edges, phi_edges, Z_edges = edges
    nR, nP, nZ = len(R_edges)-1, len(phi_edges)-1, len(Z_edges)-1

    mean = np.zeros((nR, nP, nZ), dtype=np.float64)
    M2   = np.zeros_like(mean)            # sum of squared deviations
    count = 0                             # draw counter

    # reusable buffer for per-draw counts
    buf = np.zeros((nR, nP, nZ), dtype=np.uint32)

    coords_nom = cylindrical(df)
    dist, derr = df.distance_pc.to_numpy(), df.distance_error_pc.to_numpy()

    for d in range(draws):
        buf.fill(0)                       # recycle buffer

        # jitter distances → radial scaling
        scale = rng.normal(1.0, derr / dist)
        pert  = coords_nom * scale[:, None]

        iR   = np.searchsorted(R_edges,   pert[:, 0], 'right') - 1
        iP   = np.searchsorted(phi_edges, pert[:, 1], 'right') - 1
        iZ   = np.searchsorted(Z_edges,   pert[:, 2], 'right') - 1
        ok   = (iR>=0)&(iR<nR)&(iP>=0)&(iP<nP)&(iZ>=0)&(iZ<nZ)

        np.add.at(buf, (iR[ok], iP[ok], iZ[ok]), 1)

        # ---- Welford update --------------------------------------
        count += 1
        delta = buf - mean
        mean += delta / count
        M2   += delta * (buf - mean)

        if (d+1) % max(1, draws//10) == 0:
            print(f"  · draw {d+1}/{draws}")

    # population variance → (M2 / draws); std-dev = sqrt
    sigma = np.sqrt(M2 / draws).astype(np.float32)
    return sigma


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    a = get_args()
    stars = Path(a.stars) if a.stars else auto_path(DEF_STARS, "--stars")
    cube  = Path(a.cube)  if a.cube  else auto_path(DEF_CUBE,  "--cube")
    out   = Path(a.outdir) if a.outdir else DEF_OUT
    out.mkdir(parents=True, exist_ok=True)

    df   = ensure_galactocentric(pd.read_parquet(stars))
    npz = np.load(cube, allow_pickle=True)

    # --- robust edge extraction ---------------------------------------
    if {"R_edges", "phi_edges", "Z_edges"}.issubset(npz.files):
        edges = (npz["R_edges"], npz["phi_edges"], npz["Z_edges"])
    elif "edges" in npz.files:                           # create_map_wout_spectra.py
        _e = npz["edges"]           # dtype=object, length-3
        edges = (np.asarray(_e[0]), np.asarray(_e[1]), np.asarray(_e[2]))
    else:
        sys.exit("ERROR: could not locate bin edges in voxel_cube.npz")

    print(f"↻ MC propagation: {a.draws} draws on {len(df):,} stars")
    sigma = mc_errors(df, edges, a.draws, default_rng(a.seed))

    np.savez_compressed(out / "error_cube.npz",
                        sigma=sigma,
                        R_edges=edges[0],
                        phi_edges=edges[1],
                        Z_edges=edges[2])
    print(f"✓ saved {out/'error_cube.npz'}  (shape={sigma.shape})")


if __name__ == "__main__":
    main()
