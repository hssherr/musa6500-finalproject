"""
rgb_change.py — flag likely roof collapses in Philadelphia by RGB-differencing
2024 vs 2025 source.coop aerial imagery.

Data source (confirmed April 2026):
    https://source.coop/nlebovits/phl-aerial-imagery
    - 2024 and 2025 COGs, 954 tiles/year
    - 0.25 ft / 3 inch / 0.0762 m GSD, EPSG:2272, uint8 RGB
    - STAC-GeoParquet index per year at:
        https://data.source.coop/nlebovits/phl-aerial-imagery/<year>/items.parquet
    - Individual tile URLs follow:
        https://data.source.coop/nlebovits/phl-aerial-imagery/<year>/<TILEID>.tif

Flow:
    load_tile_index(year)                -> GeoDataFrame of {url, geometry}
    per_parcel_rgb_change(parcels, ...)  -> features per parcel (diff, SSIM, dark_new)
    flag_collapses(features, ...)        -> parcels above change threshold

Dependencies:
    pip install rioxarray geopandas rasterio shapely pyarrow scikit-image tqdm
"""

from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401
from shapely.geometry import box
from tqdm import tqdm

SOURCE_COOP_BASE = "https://data.source.coop/nlebovits/phl-aerial-imagery"
PHILLY_CRS = "EPSG:2272"

# Cloudflare blocks default Python/GDAL user agents with 403. Set a
# browser-like UA before any HTTP call.
_UA = "Mozilla/5.0 (compatible; roof-collapse/1.0)"
os.environ.setdefault("GDAL_HTTP_USERAGENT", _UA)
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.TIF,.TIFF")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


# ---------------------------------------------------------------------------
# Tile index — read the STAC-GeoParquet directly
# ---------------------------------------------------------------------------
def load_tile_index(year: int, cache_dir: str | Path = "data/source_coop") -> gpd.GeoDataFrame:
    """Download and parse <year>/items.parquet. Returns a GeoDataFrame with
    columns `url` (absolute HTTPS COG URL) and `geometry` (tile footprint in
    EPSG:2272 feet).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_dir / f"items_{year}.parquet"

    if not parquet_path.exists():
        url = f"{SOURCE_COOP_BASE}/{year}/items.parquet"
        print(f"Fetching tile index: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req) as resp, open(parquet_path, "wb") as f:
            f.write(resp.read())

    import pyarrow.parquet as pq
    df = pq.read_table(parquet_path).to_pandas()

    def _row_to_record(row):
        href = row["assets"]["data"]["href"]            # e.g. "./26479E199654N.tif"
        href = href.lstrip("./")
        full_url = f"{SOURCE_COOP_BASE}/{year}/{href}"
        xmin, ymin, xmax, ymax = row["proj:bbox"]       # EPSG:2272 feet
        return {"url": full_url, "geometry": box(xmin, ymin, xmax, ymax)}

    records = [_row_to_record(r) for _, r in df.iterrows()]
    return gpd.GeoDataFrame(records, crs=PHILLY_CRS)


# ---------------------------------------------------------------------------
# Chip extraction
# ---------------------------------------------------------------------------
def _window_read(tile_url: str, bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Read an (H, W, 3) uint8 chip from a COG over HTTP by EPSG:2272 bounds."""
    vsi = f"/vsicurl/{tile_url}"
    minx, miny, maxx, maxy = bounds
    with rasterio.open(vsi) as src:
        win = src.window(minx, miny, maxx, maxy)
        arr = src.read(indexes=[1, 2, 3], window=win, boundless=True, fill_value=0)
    return np.transpose(arr, (1, 2, 0))


def chip_pair(
    parcel_geom,
    tile_index_a: gpd.GeoDataFrame,
    tile_index_b: gpd.GeoDataFrame,
    buffer_ft: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    minx, miny, maxx, maxy = parcel_geom.buffer(buffer_ft).bounds
    bounds = (minx, miny, maxx, maxy)
    parcel_box = box(*bounds)

    hit_a = tile_index_a[tile_index_a.intersects(parcel_box)]
    hit_b = tile_index_b[tile_index_b.intersects(parcel_box)]
    if hit_a.empty or hit_b.empty:
        return np.zeros((0, 0, 3), np.uint8), np.zeros((0, 0, 3), np.uint8)

    a = _window_read(hit_a.iloc[0]["url"], bounds)
    b = _window_read(hit_b.iloc[0]["url"], bounds)

    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


# ---------------------------------------------------------------------------
# Change features
# ---------------------------------------------------------------------------
def rgb_change_features(chip_a: np.ndarray, chip_b: np.ndarray) -> dict:
    if chip_a.size == 0 or chip_b.size == 0 or chip_a.shape != chip_b.shape:
        return {
            "mean_abs_diff": np.nan, "p95_abs_diff": np.nan,
            "ssim": np.nan, "dark_new_frac": np.nan,
            "pixel_count": 0,
        }
    a = chip_a.astype(np.float32) / 255.0
    b = chip_b.astype(np.float32) / 255.0
    abs_diff = np.abs(a - b).mean(axis=-1)

    try:
        from skimage.metrics import structural_similarity as ssim
        ga = a.mean(axis=-1); gb = b.mean(axis=-1)
        s = float(ssim(ga, gb, data_range=1.0))
    except Exception:
        s = float("nan")

    lum_a = a.mean(axis=-1); lum_b = b.mean(axis=-1)
    dark_new = (lum_a > 0.30) & (lum_b < 0.15)

    return {
        "mean_abs_diff": float(abs_diff.mean()),
        "p95_abs_diff":  float(np.percentile(abs_diff, 95)),
        "ssim":          s,
        "dark_new_frac": float(dark_new.mean()),
        "pixel_count":   int(a.shape[0] * a.shape[1]),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def per_parcel_rgb_change(
    parcels: gpd.GeoDataFrame,
    tile_index_a: gpd.GeoDataFrame,
    tile_index_b: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    parcels = parcels.to_crs(PHILLY_CRS).reset_index(drop=True)
    feats: list[dict] = []
    for _, row in tqdm(parcels.iterrows(), total=len(parcels), desc="RGB diff"):
        a, b = chip_pair(row.geometry, tile_index_a, tile_index_b)
        feats.append(rgb_change_features(a, b))
    feats_df = pd.DataFrame(feats)
    return gpd.GeoDataFrame(
        pd.concat([parcels, feats_df], axis=1),
        geometry=parcels.geometry,
        crs=PHILLY_CRS,
    )


def flag_collapses(
    features: gpd.GeoDataFrame,
    min_mean_abs_diff: float = 0.18,
    min_dark_new_frac: float = 0.05,
    max_ssim: float = 0.55,
) -> gpd.GeoDataFrame:
    """A parcel is flagged if pixel-level change is large AND either a chunk
    of newly-dark pixels (exposed cavity) OR low SSIM (structural change).
    Calibrate against 10–20 hand-labeled collapses before trusting defaults.
    """
    f = features.copy()
    cond = (
        (f["mean_abs_diff"].fillna(0) >= min_mean_abs_diff)
        & (
            (f["dark_new_frac"].fillna(0) >= min_dark_new_frac)
            | (f["ssim"].fillna(1.0) <= max_ssim)
        )
    )
    flagged = f.loc[cond].copy()
    flagged["change_score"] = (
        flagged["mean_abs_diff"].fillna(0) * 0.5
        + flagged["dark_new_frac"].fillna(0) * 0.3
        + (1 - flagged["ssim"].fillna(1.0)) * 0.2
    )
    return flagged.sort_values("change_score", ascending=False)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def main(
    parcels_geojson: str,
    year_a: int = 2024,
    year_b: int = 2025,
    out_dir: str = "data/rgb_change",
):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    idx_a = load_tile_index(year_a)
    idx_b = load_tile_index(year_b)
    idx_a.to_file(out_dir / f"tiles_{year_a}.geojson", driver="GeoJSON")
    idx_b.to_file(out_dir / f"tiles_{year_b}.geojson", driver="GeoJSON")
    print(f"Tile index: {len(idx_a)} tiles in {year_a}, {len(idx_b)} in {year_b}")

    parcels = gpd.read_file(parcels_geojson)
    features = per_parcel_rgb_change(parcels, idx_a, idx_b)
    features.to_file(out_dir / "parcel_rgb_change.geojson", driver="GeoJSON")

    flagged = flag_collapses(features)
    flagged.to_file(out_dir / "flagged_rgb.geojson", driver="GeoJSON")
    print(f"\nFlagged {len(flagged):,} parcels. Top 20 by change score:")
    print(flagged.head(20)[
        ["mean_abs_diff", "p95_abs_diff", "ssim", "dark_new_frac", "change_score"]
    ])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--parcels", required=True)
    ap.add_argument("--year-a", type=int, default=2024)
    ap.add_argument("--year-b", type=int, default=2025)
    ap.add_argument("--out-dir", default="data/rgb_change")
    args = ap.parse_args()

    main(args.parcels, year_a=args.year_a, year_b=args.year_b, out_dir=args.out_dir)
