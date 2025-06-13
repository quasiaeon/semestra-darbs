#!/usr/bin/env python
"""
create_map.py — Cylindrical 3-D Density & Uncertainty Map (no spectra)

• Looks for  data/stars_clean.parquet  when no --input is given.
• Adds Galactocentric (x,y,z) if missing.
• Builds a cylindrical voxel cube  (0.5 kpc × 0.5° × 0.5 kpc).
• Propagates parallax errors via Monte-Carlo (default 1000 draws).
• Saves:
    output/voxel_cube.npz   – log_density + edges
    output/error_cube.npz   – sigma(N)    + edges
    output/milky_way_map.html
    output/polar_slice.html
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go, plotly.express as px, plotly.io as pio
from numpy.random import default_rng
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u

# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/stars_clean.parquet",
                   help="Parquet with RA, Dec, distance_pc "
                        "(and optionally x,y,z columns)")
    p.add_argument("--outdir", default="output",
                   help="Destination folder (default: output/)")
    p.add_argument("--draws",  type=int, default=1000,
                   help="Monte-Carlo realisations for σ(N) [1000]")
    p.add_argument("--seed",   type=int, default=42,
                   help="Random seed [42]")
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ────────────────────────────────────────────────────────────────────
def ensure_galactocentric(df: pd.DataFrame) -> pd.DataFrame:
    """Add x,y,z [kpc] columns if missing."""
    if {"x", "y", "z"}.issubset(df.columns):
        return df
    print("→ Deriving Galactocentric x, y, z via Astropy …")
    c_icrs = SkyCoord(ra=df.ra.values * u.deg,
                      dec=df.dec.values * u.deg,
                      distance=df.distance_pc.values * u.pc,
                      frame="icrs")
    gal = c_icrs.transform_to(Galactocentric)
    df["x"] = gal.x.to_value(u.kpc)
    df["y"] = gal.y.to_value(u.kpc)
    df["z"] = gal.z.to_value(u.kpc)
    return df

def cylindrical_RphiZ(df: pd.DataFrame):
    R   = np.hypot(df.x.values, df.y.values)
    phi = np.degrees(np.arctan2(df.y.values, df.x.values))
    return R, phi, df.z.values

# ────────────────────────────────────────────────────────────────────
# Voxel cube
# ────────────────────────────────────────────────────────────────────
def density_cube(R, phi, z):
    R_edges   = np.arange(0, 100.5, 0.5)     # 0.5 kpc
    phi_edges = np.arange(-180, 180.5, 0.5)  # 0.5°
    z_edges   = np.arange(-15,  15.5, 0.5)   # 0.5 kpc
    H, _ = np.histogramdd(np.column_stack([R, phi, z]),
                          bins=[R_edges, phi_edges, z_edges])
    return H, (R_edges, phi_edges, z_edges)

# ────────────────────────────────────────────────────────────────────
# Monte-Carlo σ(N)  (memory-light streaming)
# ────────────────────────────────────────────────────────────────────
def mc_sigma(df: pd.DataFrame, edges, draws: int, seed: int):
    if "distance_error_pc" not in df.columns:
        print("⚠  distance_error_pc missing – skipping σ(N)")
        return np.zeros(tuple(len(e)-1 for e in edges), np.float32)

    rng   = default_rng(seed)
    nR, nP, nZ = (len(e)-1 for e in edges)
    mean  = np.zeros((nR, nP, nZ), np.float64)
    M2    = np.zeros_like(mean)
    buf   = np.zeros_like(mean, dtype=np.uint32)  # reusable per draw

    coords = np.column_stack(cylindrical_RphiZ(df))  # R,phi,z
    dist   = df.distance_pc.values
    derr   = df.distance_error_pc.values

    for d in range(draws):
        buf.fill(0)
        scale = rng.normal(1.0, derr / dist)
        pert  = coords * scale[:, None]

        iR   = np.searchsorted(edges[0], pert[:, 0], "right") - 1
        iP   = np.searchsorted(edges[1], pert[:, 1], "right") - 1
        iZ   = np.searchsorted(edges[2], pert[:, 2], "right") - 1
        ok   = (iR>=0)&(iR<nR)&(iP>=0)&(iP<nP)&(iZ>=0)&(iZ<nZ)
        np.add.at(buf, (iR[ok], iP[ok], iZ[ok]), 1)

        # Welford online update
        delta = buf - mean
        mean += delta / (d + 1)
        M2   += delta * (buf - mean)

        if (d+1) % max(1, draws//10) == 0:
            print(f"  · draw {d+1}/{draws}")

    return np.sqrt(M2 / draws).astype(np.float32)

# ────────────────────────────────────────────────────────────────────
# Plotting helpers
# ────────────────────────────────────────────────────────────────────
def plot_volume(H, sig, edges, out_html: Path):
    logH = np.where(H > 0, np.log10(H), np.nan)
    finite = np.isfinite(logH)
    if not finite.any():
        print("⚠  No voxels > 0; skipping volume plot.")
        return
    vmin, vmax = np.nanpercentile(logH[finite], [15, 99])

    Rc   = 0.5 * (edges[0][:-1] + edges[0][1:])
    phic = 0.5 * (edges[1][:-1] + edges[1][1:])
    zc   = 0.5 * (edges[2][:-1] + edges[2][1:])
    X, Y, Z = np.meshgrid(Rc, phic, zc, indexing="ij")

    fig = go.Figure(go.Volume(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        value=logH.ravel(),
        customdata=sig.ravel(),
        isomin=vmin, isomax=vmax,
        opacity=0.10, surface_count=15,
        coloraxis="coloraxis",
        hovertemplate="log₁₀(N): %{value:.2f}<br>σ: %{customdata:.1f}"
    ))
    fig.update_layout(
        title="Milky Way log-density with 1 σ uncertainty",
        coloraxis=dict(colorscale="Viridis"),
        scene=dict(xaxis_title="R [kpc]",
                   yaxis_title="φ [deg]",
                   zaxis_title="Z [kpc]")
    )
    pio.write_html(fig, out_html)
    print(f"✓ wrote {out_html}")

def plot_polar_slice(df: pd.DataFrame, out_html: Path):
    thin = df[np.abs(df.z) < 0.3]
    R, phi, _ = cylindrical_RphiZ(thin)
    fig = px.scatter_polar(thin, r=R, theta=phi, opacity=0.4,
                           title="Thin-disc face-on slice (|z|<0.3 kpc)",
                           color=R, color_continuous_scale="Viridis")
    pio.write_html(fig, out_html)
    print(f"✓ wrote {out_html}")

# ────────────────────────────────────────────────────────────────────
def main():
    a = get_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if not Path(a.input).is_file():
        sys.exit(f"ERROR: {a.input} not found.")
    df = pd.read_parquet(a.input)
    print(f"Loaded {len(df):,} stars")

    df = ensure_galactocentric(df)
    R, phi, z = cylindrical_RphiZ(df)
    H, edges  = density_cube(R, phi, z)

    # ── save density cube ──────────────────────────────────────────
    np.savez_compressed(outdir / "voxel_cube.npz",
                        log_density=np.where(H>0, np.log10(H), np.nan),
                        edges=np.array(edges, dtype=object))
    print(f"✓ wrote {outdir/'voxel_cube.npz'}")

    # ── Monte-Carlo σ(N) ───────────────────────────────────────────
    print(f"↻ Monte-Carlo σ(N), {a.draws} draws …")
    sig = mc_sigma(df, edges, draws=a.draws, seed=a.seed)
    np.savez_compressed(outdir / "error_cube.npz",
                        sigma=sig, edges=np.array(edges, dtype=object))
    print(f"✓ wrote {outdir/'error_cube.npz'}")

    # ── plots ──────────────────────────────────────────────────────
    plot_volume(H, sig, edges, outdir / "milky_way_map.html")
    plot_polar_slice(df, outdir / "polar_slice.html")

if __name__ == "__main__":
    main()