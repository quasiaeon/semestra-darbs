"""Download a sky subset from Gaia DR3 using *only* astroquery.

Examples
--------
# 1 million bright stars in a 10°×10° field centred on the Galactic plane
python download_data.py \
    --ra 180 190 --dec -5 5 \
    --gmag-max 17 --rows 1000000 \
    --out-dir data

Outputs
-------
- data/stars_raw.parquet   ← raw gaia_source columns
- data/xp_raw.parquet      ← optional BP/RP spectra (one row per star)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from astroquery.gaia import Gaia

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # explicit for clarity

# ---------- helpers ---------------------------------------------------------

def adql_gaia_source(ra_min: float, ra_max: float,
                      dec_min: float, dec_max: float,
                      gmag_max: float, limit: int) -> str:
    """Return an ADQL query for a rectangular sky cut."""
    return f"""
    SELECT
        gs.source_id, gs.ra, gs.dec,
        gs.parallax, gs.parallax_error, gs.parallax_over_error,
        gs.pmra, gs.pmdec, gs.radial_velocity,
        gs.phot_g_mean_mag, gs.bp_rp
    FROM gaiadr3.gaia_source AS gs
    WHERE gs.ra BETWEEN {ra_min} AND {ra_max}
      AND gs.dec BETWEEN {dec_min} AND {dec_max}
      AND gs.phot_g_mean_mag < {gmag_max}
    LIMIT {limit}
    """

def adql_xp_sampled(source_ids: list[int]) -> str:
    """Return ADQL for XP sampled spectra for a *small* list of source_ids."""
    # ADQL has a  prepared statement size limit; chunk long lists upstream.
    id_list = ",".join(map(str, source_ids))
    return f"""
    SELECT xps.source_id, xps.wave, xps.flux
    FROM gaiadr3.xp_sampled_mean_spectrum AS xps
    WHERE xps.source_id IN ({id_list})
    """

def fetch_to_dataframe(adql: str):
    job = Gaia.launch_job_async(adql)
    return job.get_results().to_pandas()

# ---------- CLI -------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("Gaia DR3 downloader (astroquery‑only)")
    p.add_argument("--ra", nargs=2, type=float, metavar=("MIN", "MAX"), required=True,
                   help="RA range in degrees (ICRS)")
    p.add_argument("--dec", nargs=2, type=float, metavar=("MIN", "MAX"), required=True,
                   help="Dec range in degrees (ICRS)")
    p.add_argument("--gmag-max", type=float, default=17.0,
                   help="Bright‑side G‑mag cut [default: 17]")
    p.add_argument("--rows", type=int, default=100_000,
                   help="Maximum number of rows to fetch [default: 1e5]")
    p.add_argument("--spectra", action="store_true",
                   help="Also download BP/RP XP spectra for the same stars (⚠ slows query)")
    p.add_argument("--out-dir", default="data", help="Destination directory [data]")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. gaia_source subset --------------------------------------------------
    adql = adql_gaia_source(*args.ra, *args.dec, args.gmag_max, args.rows)
    df_star = fetch_to_dataframe(adql)
    df_star.to_parquet(out / "stars_raw.parquet", index=False)
    print(f"✓ Wrote {len(df_star):,} rows to stars_raw.parquet")

    # 2. optional XP spectra --------------------------------------------------
    if args.spectra:
        # Split into <= 50 000‑ID chunks to keep ADQL manageable
        chunks = [df_star.source_id.iloc[i:i+50_000].tolist()
                  for i in range(0, len(df_star), 50_000)]
        df_xp = pd.concat([fetch_to_dataframe(adql_xp_sampled(c)) for c in chunks])
        df_xp.to_parquet(out / "xp_raw.parquet", index=False)
        print(f"✓ Wrote {len(df_xp):,} spectra to xp_raw.parquet")

if __name__ == "__main__":
    main()