#!/usr/bin/env python
"""
download_data.py  –  Flexible Gaia-DR3 downloader (ICRS or Galactic cuts)

Outputs
-------
data/stars_raw.parquet    – scalar Gaia columns
data/xp_raw.parquet       – BP/RP sampled spectra (optional)

Examples
--------
# 1) All-sky |b|<20° slice, G<18, split into 30° longitude chunks
python download_data.py --gal-l 0 360 --gal-b -20 20 --gmag-max 18

# 2) Specific RA/Dec rectangle (legacy mode)
python download_data.py --ra 170 190 --dec -10 10 --gmag-max 17.5
"""
from __future__ import annotations
import argparse, textwrap, time, random, sys
from pathlib import Path
import numpy as np, pandas as pd
from astroquery.gaia import Gaia
from astroquery.exceptions import RemoteServiceError
from requests.exceptions  import HTTPError
import pyvo

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1                             # disable client row-cap
ESA_TAP = "https://gea.esac.esa.int/tap-server/tap"
SJS_BASE = "https://gaia.aip.de/uws/simple-join-service"

# --------------------------------------------------------------------------- #
# ADQL helpers
# --------------------------------------------------------------------------- #
COLS = ("source_id, ra, dec, l, b, "
        "parallax, parallax_error, phot_g_mean_mag, bp_rp, ruwe")

def q_icrs(ra_min, ra_max, dec_min, dec_max, gmax, top) -> str:
    return textwrap.dedent(f"""
        SELECT { 'TOP '+str(top) if top>0 else '' }
          {COLS}
        FROM gaiadr3.gaia_source
        WHERE ra  BETWEEN {ra_min}  AND {ra_max}
          AND dec BETWEEN {dec_min} AND {dec_max}
          AND phot_g_mean_mag < {gmax}
    """)

def q_gal(l_min, l_max, b_min, b_max, gmax, top) -> str:
    return textwrap.dedent(f"""
        SELECT { 'TOP '+str(top) if top>0 else '' }
          {COLS}
        FROM gaiadr3.gaia_source
        WHERE l BETWEEN {l_min} AND {l_max}
          AND b BETWEEN {b_min} AND {b_max}
          AND phot_g_mean_mag < {gmax}
    """)

def tap_to_df(adql: str, fmt="csv", tries=3) -> pd.DataFrame:
    last = None
    for _ in range(tries):
        try:
            job = Gaia.launch_job_async(adql, output_format=fmt,
                                         dump_to_file=False)
            return job.get_results().to_pandas()
        except (RemoteServiceError, HTTPError, OSError, ValueError) as err:
            last = err
            time.sleep(1 + random.random()*2)
    raise RuntimeError(f"TAP query failed: {last}")

# --------------------------------------------------------------------------- #
# XP spectra helper (unchanged)
# --------------------------------------------------------------------------- #
def fetch_xp_aip(source_ids: list[int]) -> pd.DataFrame:
    if len(source_ids) > 2000:
        raise ValueError("SJS call limited to 2000 source_ids")
    tap = pyvo.dal.TAPService(SJS_BASE)
    sql = f"""
        SELECT source_id, wave, flux
        FROM gaiadr3.xp_sampled_mean_spectrum
        WHERE source_id IN ({','.join(map(str, source_ids))})
    """
    job = tap.submit_job(sql, language="pg-sql", format="csv", runid="gaia-xp")
    job.run(); job.wait(phases=["COMPLETED", "ERROR"], timeout=300)
    if job.phase == "ERROR":
        raise RuntimeError(job.error_summary)
    return job.fetch_result().to_table().to_pandas()

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser("Gaia DR3 downloader")
    # ICRS
    p.add_argument("--ra",  nargs=2, type=float, metavar=("MIN", "MAX"))
    p.add_argument("--dec", nargs=2, type=float, metavar=("MIN", "MAX"))
    # Galactic
    p.add_argument("--gal-l", nargs=2, type=float, metavar=("MIN", "MAX"))
    p.add_argument("--gal-b", nargs=2, type=float, metavar=("MIN", "MAX"))
    # Common
    p.add_argument("--gmag-max", type=float, default=18.0)
    p.add_argument("--rows", type=int, default=-1,
                   help="Row cap per query (-1 = no TOP clause)")
    p.add_argument("--spectra", action="store_true", help="also fetch XP spectra")
    p.add_argument("--outdir", default="data")
    # chunking for galactic mode
    p.add_argument("--chunk-size", type=float, default=30.0,
                   help="Longitude chunk size in degrees (Galactic mode)")
    return p.parse_args()

# --------------------------------------------------------------------------- #
def main():
    a = get_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    icrs_mode = a.ra and a.dec
    gal_mode  = a.gal_l and a.gal_b
    if icrs_mode and gal_mode:
        sys.exit("ERROR: Specify either RA/Dec or l/b, not both.")
    if not (icrs_mode or gal_mode):
        sys.exit("ERROR: You must provide either RA/Dec or l/b cuts.")

    # -------- ICRS rectangle -------------------------------------------------
    if icrs_mode:
        print("⭳ ICRS rectangle …")
        df = tap_to_df(q_icrs(*a.ra, *a.dec, a.gmag_max, a.rows))
        if df.empty:
            sys.exit("No stars returned — adjust sky region or magnitude cut.")
        df.to_parquet(outdir / "stars_raw.parquet", index=False)
        print(f"✓ {len(df):,} stars → stars_raw.parquet")

    # -------- Galactic latitude slice (chunked in l) ------------------------
    else:
        l0, l1 = a.gal_l
        b0, b1 = a.gal_b
        if l1 <= l0: l1 += 360        # allow wrap-around
        frames = []
        chunk = a.chunk_size
        total = 0
        for L in np.arange(l0, l1, chunk):
            L_hi = min(L + chunk, l1)
            print(f"⭳ chunk l=[{L:.0f},{L_hi:.0f}]° …")
            adql = q_gal(L, L_hi, b0, b1, a.gmag_max, a.rows)
            part = tap_to_df(adql)
            total += len(part)
            print(f"  · {len(part):,} rows")
            frames.append(part)
        if not frames:
            sys.exit("No stars returned — check l/b range.")
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(outdir / "stars_raw.parquet", index=False)
        print(f"✓ {total:,} stars → stars_raw.parquet")

    # ---- optional XP spectra ----------------------------------------------
    if a.spectra and not df.empty:
        print("⭳ XP spectra via Gaia@AIP …")
        CHUNK = 2000
        frames = []
        ids = df.source_id.tolist()
        for i in range(0, len(ids), CHUNK):
            sub = ids[i:i+CHUNK]
            print(f"  · chunk {i//CHUNK+1}: {len(sub)} IDs")
            frames.append(fetch_xp_aip(sub))
        df_xp = pd.concat(frames, ignore_index=True)
        df_xp.to_parquet(outdir / "xp_raw.parquet", index=False)
        print(f"✓ {len(df_xp):,} spectra → xp_raw.parquet")

if __name__ == "__main__":
    main()