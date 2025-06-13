# create_map.py – Cylindrical 3‑D Mapping (auto‑input v1.1)
# =============================================================
# Now *zero‑config*: if you run the script with **no arguments** it will
# look for the canonical filenames produced by earlier stages:
#   data/stars_clean.parquet          (required)
#   data/spectra_clean.parquet        (optional; enables colour‑by‑index)
# You can still override paths via --input / --spectra.

from __future__ import annotations
import argparse, sys, json, gzip
from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
import pyvo
import pandas as pd
import time, io

DEFAULT_INPUT   = r"C:\Users\atsve\Desktop\work\data\stars_clean.parquet"
DEFAULT_SPECTRA = r"C:\Users\atsve\Desktop\work\data\spectra_clean.parquet"
DEFAULT_OUTDIR  = "output"

# -------------------------------------------------------------- CLI -------

def get_args():
    p = argparse.ArgumentParser("Gaia cylindrical map builder (auto‑input)")
    p.add_argument("--input",   default=DEFAULT_INPUT, help="Parquet file from process_data.py")
    p.add_argument("--spectra", default=DEFAULT_SPECTRA, help="Optional spectra_clean.parquet")
    p.add_argument("--outdir",  default="DEFAULT_OUTDIR",  help="Destination directory")
    p.add_argument("--voxel-R", type=float, default=0.1, help="Radial bin [kpc]")
    p.add_argument("--voxel-Z", type=float, default=0.1, help="Vertical bin [kpc]")
    p.add_argument("--voxel-phi", type=float, default=1.0, help="Azimuth bin [deg]")
    return p.parse_args()

args = get_args()

# ----------- smart defaults ---------------------------------------------
root = Path(__file__).resolve().parent

if args.input is None:
    auto = root / "data" / "stars_clean.parquet"
    if auto.exists():
        args.input = auto
        print(f"[auto] using {auto} as --input")
    else:
        sys.exit("ERROR: --input not given and data/stars_clean.parquet not found.")

if args.spectra is None:
    auto = root / "data" / "spectra_clean.parquet"
    if auto.exists():
        args.spectra = auto
        print(f"[auto] using {auto} as --spectra")

outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------ load data --
print("⭳ reading", args.input)
df = pd.read_parquet(args.input, engine="pyarrow")

has_spec = args.spectra is not None
if has_spec:
    print("⭳ reading", args.spectra)
    spec = pd.read_parquet(args.spectra, engine="pyarrow")
    df = df.merge(spec[["source_id", "mg_idx"]], on="source_id", how="left")

# ------------------------------------------------ cylindrical transform --
R = np.hypot(df.x, df.y)   # x,y were written by process_data (Galactocentric)
phi = np.degrees(np.arctan2(df.y, df.x))

# ------------------------------------------------ histogram --------------
R_edges   = np.arange(0, 30+args.voxel_R, args.voxel_R)
phi_edges = np.arange(-180, 180+args.voxel_phi, args.voxel_phi)
Z_edges   = np.arange(-4, 4+args.voxel_Z, args.voxel_Z)

coords = np.column_stack((R, phi, df.z))
print("⎈ voxelising …")
H, _ = np.histogramdd(coords, bins=[R_edges, phi_edges, Z_edges])

np.savez_compressed(outdir/"voxel_cube.npz", density=H, R_edges=R_edges,
                    phi_edges=phi_edges, Z_edges=Z_edges)
print("✓ voxel_cube.npz written")

# -------------- polar slice for |z|<0.3 kpc ------------------------------
sel = df.z.abs() < 0.3
scatter = dict(mode="markers", marker=dict(size=1, opacity=0.3,
                  color=df.mg_idx[sel] if has_spec else R[sel],
                  colorscale="Turbo", showscale=has_spec))
fig = go.Figure(go.Scatterpolar(r=R[sel], theta=phi[sel], **scatter))
fig.update_layout(template="plotly_dark", title="Face‑on Thin‑Disc Slice")
fig.write_html(outdir/"polar_slice.html", include_plotlyjs="cdn")
print("✓ polar_slice.html written")

# -------------- 3‑D volume ----------------------------------------------
import plotly.express as px
vol = px.volume(np.log10(H+1), x=R_edges[:-1], y=phi_edges[:-1], z=Z_edges[:-1],
                title="Galactic Density (log₁₀)")
vol.write_html(outdir/"milky_way_map.html", include_plotlyjs="cdn")
print("✓ milky_way_map.html written")