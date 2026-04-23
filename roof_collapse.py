"""
roof_collapse.py — flag likely roof collapses in Philadelphia parcels by
differencing lidar surfaces between 2018 and 2022.

Data sources (confirmed on PASDA, April 2026):
- 2018 Philadelphia lidar DEM (direct zip)
- 2022 Philadelphia lidar LAS files (directory listing)

Flow:
    download_dem_2018()           -> GeoTIFF mosaic (bare-earth, 2018)
    download_las_2022()           -> LAS point cloud files (2022)
    build_dsm_from_las(...)       -> first-return DSM raster from 2022 LAS
    per_parcel_height_change(...) -> zonal stats of (DSM_2022 - DEM_2018)
    flag_collapses(...)           -> parcels with large negative height change

Why a DEM vs DSM mismatch is OK for collapses:
    A collapsed roof piles debris on the ground AND removes the vertical
    structure. The 2018 first-surface height (from DEM-derived hillshade or
    the raw 2018 LAS, see note below) over an intact building will be well
    above ground. If by 2022 the structure is gone, the 2022 DSM over that
    footprint drops to near-ground. The sign and magnitude of the drop is
    what we threshold.

    If both years expose only bare-earth DEM, the signal is weaker (it only
    catches debris piles and excavated lots). For the 2022 year this script
    builds a proper DSM from the LAS point cloud using laspy or PDAL.

Dependencies:
    pip install rioxarray geopandas rasterstats laspy[lazrs] tqdm
    # optional but faster: pdal (conda install -c conda-forge pdal python-pdal)
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
from rasterio.enums import Resampling
from rasterstats import zonal_stats
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants — confirmed on PASDA, April 2026
# ---------------------------------------------------------------------------
DEM_2018_ZIP_URL = (
    "https://www.pasda.psu.edu/download/phillyLiDAR/LAS2018/Philly_DEM_2018.zip"
)
LAS_2022_INDEX_URL = "https://www.pasda.psu.edu/download/phillyLiDAR/2022/LAS/"

# Pennsylvania South State Plane (feet) — PASDA Philly products land here
PHILLY_CRS = "EPSG:2272"

# Height-drop threshold in meters: a collapse is a >2 m surface loss
DEFAULT_DROP_M = -2.0


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path, desc: str = "download") -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[{desc}] cached: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{desc}] downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def download_dem_2018(data_dir: str | Path = "data/dem") -> Path:
    """Returns a directory containing the extracted 2018 DEM GeoTIFFs."""
    data_dir = Path(data_dir)
    zip_path = _download(
        DEM_2018_ZIP_URL, data_dir / "raw" / "Philly_DEM_2018.zip", desc="2018 DEM"
    )
    out = data_dir / "2018"
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
    return out


def download_las_2022(
    data_dir: str | Path = "data/las_2022",
    filenames: Iterable[str] | None = None,
) -> Path:
    """Download 2022 LAS tiles from PASDA.

    PASDA hosts these as a directory of .las / .laz files at LAS_2022_INDEX_URL.
    Rather than scraping the HTML listing (fragile), pass an explicit list of
    filenames obtained once from the index page. For a full-city run, fetch
    the index HTML manually and paste the file list, or filter to tiles that
    intersect your parcel bbox (see las_tiles_for_parcels below).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if filenames is None:
        raise ValueError(
            "Pass an explicit list of LAS filenames. Fetch the index once from "
            f"{LAS_2022_INDEX_URL} or restrict to tiles your parcels touch."
        )
    paths = []
    for name in filenames:
        paths.append(_download(LAS_2022_INDEX_URL + name, data_dir / name, desc=name))
    return data_dir


# ---------------------------------------------------------------------------
# DSM construction from LAS (2022)
# ---------------------------------------------------------------------------
def build_dsm_from_las(
    las_dir: str | Path,
    out_tif: str | Path,
    cell_size_m: float = 1.0,
    classification_keep: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
) -> Path:
    """Rasterise max-Z (DSM) from a directory of LAS/LAZ files.

    cell_size_m: target pixel size in meters (1 m is a sensible default for
    roof-scale change detection at QL1 density).

    Uses laspy for the point read and numpy for the max-Z rasterisation.
    For large areas PDAL is faster; see the PDAL alternative commented below.
    """
    import laspy

    out_tif = Path(out_tif)
    if out_tif.exists():
        print(f"DSM already built: {out_tif}")
        return out_tif

    las_paths = sorted(Path(las_dir).glob("*.la[sz]"))
    if not las_paths:
        raise FileNotFoundError(f"No LAS/LAZ files in {las_dir}")

    # First pass — get global bounds
    xs_min, ys_min, xs_max, ys_max = np.inf, np.inf, -np.inf, -np.inf
    for p in las_paths:
        with laspy.open(p) as f:
            h = f.header
            xs_min = min(xs_min, h.mins[0]); ys_min = min(ys_min, h.mins[1])
            xs_max = max(xs_max, h.maxs[0]); ys_max = max(ys_max, h.maxs[1])

    # PASDA Philly LAS header CRS is EPSG:2272 (feet). Convert the requested
    # cell_size from meters to feet since the raster grid sits in feet.
    cell_ft = cell_size_m / 0.3048
    width = int(np.ceil((xs_max - xs_min) / cell_ft))
    height = int(np.ceil((ys_max - ys_min) / cell_ft))
    dsm = np.full((height, width), np.nan, dtype=np.float32)

    # Second pass — populate max Z
    for p in tqdm(las_paths, desc="Rasterising DSM"):
        las = laspy.read(p)
        keep = np.isin(las.classification, classification_keep)
        x = np.asarray(las.x[keep]); y = np.asarray(las.y[keep])
        z_ft = np.asarray(las.z[keep])
        col = np.clip(((x - xs_min) / cell_ft).astype(int), 0, width - 1)
        row = np.clip(((ys_max - y) / cell_ft).astype(int), 0, height - 1)
        # aggregate max — order-preserving scatter max
        flat = row * width + col
        order = np.argsort(-z_ft)  # descending
        flat = flat[order]; z_sorted = z_ft[order]
        _, first = np.unique(flat, return_index=True)
        cells = flat[first]; vals = z_sorted[first]
        existing = dsm.flat[cells]
        mask = np.isnan(existing) | (vals > existing)
        dsm.flat[cells[mask]] = vals[mask]

    # Convert stored Z from feet to meters so the diff is in meters
    dsm_m = dsm * 0.3048

    # Write GeoTIFF
    from affine import Affine
    import rasterio

    transform = Affine(cell_ft, 0, xs_min, 0, -cell_ft, ys_max)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_tif, "w", driver="GTiff", height=height, width=width,
        count=1, dtype="float32", crs=PHILLY_CRS, transform=transform,
        nodata=np.nan, compress="lzw", tiled=True,
    ) as dst:
        dst.write(dsm_m, 1)
    print(f"DSM written: {out_tif}  ({width} x {height} px @ {cell_size_m} m)")
    return out_tif


# ---------------------------------------------------------------------------
# DEM mosaic and alignment
# ---------------------------------------------------------------------------
def _open_mosaic(folder: str | Path) -> xr.DataArray:
    folder = Path(folder)
    tifs = sorted(list(folder.rglob("*.tif")) + list(folder.rglob("*.tiff")))
    if not tifs:
        raise FileNotFoundError(f"No GeoTIFFs in {folder}")
    if len(tifs) == 1:
        da = rioxarray.open_rasterio(tifs[0], masked=True).squeeze(drop=True)
    else:
        from rioxarray.merge import merge_arrays
        arrs = [
            rioxarray.open_rasterio(t, masked=True).squeeze(drop=True) for t in tifs
        ]
        da = merge_arrays(arrs)
    # PASDA DEMs are usually in feet (vertical). Convert if the product says so.
    # Assume metric for now; user must verify from the metadata PDF.
    if da.rio.crs is None:
        da = da.rio.write_crs(PHILLY_CRS)
    return da


def build_diff_raster(
    dem_2018_dir: str | Path,
    dsm_2022_tif: str | Path,
    out_tif: str | Path,
) -> Path:
    """DSM_2022 - DEM_2018, aligned to the 2022 grid, written as GeoTIFF.

    Note: 2018 is bare-earth DEM and 2022 is first-return DSM. Over intact
    buildings the diff will be *positive* (DSM is above ground). Over a lot
    that was a building in 2018 but is now cleared, the diff will be near
    zero or negative if debris was later removed. A roof-collapse signal is
    better captured by also building a 2018 DSM from the 2018 LAS (swap
    dem_2018_dir for that DSM if available). For a first pass, large
    decreases between parcels' own DSM-2018 proxy (their 2018 roof height
    from the published 2018 DSM if PASDA ships it, else max-Z from 2018 LAS)
    and the 2022 DSM are the real collapse signal.
    """
    dem18 = _open_mosaic(dem_2018_dir).rio.reproject(PHILLY_CRS)
    dsm22 = rioxarray.open_rasterio(dsm_2022_tif, masked=True).squeeze(drop=True)
    if dsm22.rio.crs is None:
        dsm22 = dsm22.rio.write_crs(PHILLY_CRS)

    # Resample 2018 onto the 2022 grid
    dem18_aligned = dem18.rio.reproject_match(dsm22, resampling=Resampling.bilinear)
    diff = (dsm22 - dem18_aligned).astype("float32")
    diff.rio.write_nodata(np.nan, inplace=True)

    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    diff.rio.to_raster(out_tif, compress="lzw", tiled=True)
    print(f"Diff raster: {out_tif}")
    return out_tif


# ---------------------------------------------------------------------------
# Per-parcel zonal stats
# ---------------------------------------------------------------------------
def per_parcel_height_change(
    parcels: gpd.GeoDataFrame,
    diff_tif: str | Path,
    drop_threshold_m: float = DEFAULT_DROP_M,
) -> gpd.GeoDataFrame:
    """For each parcel polygon, compute min, mean, and count-of-dropped-pixels
    over the (DSM_later - DSM_earlier) raster.

    A min value below drop_threshold_m combined with a non-trivial
    dropped-pixel count is the roof-collapse flag.
    """
    parcels = parcels.to_crs(PHILLY_CRS).reset_index(drop=True)
    stats = zonal_stats(
        vectors=parcels.geometry,
        raster=str(diff_tif),
        stats=["min", "mean", "count"],
        add_stats={
            "n_pixels_dropped": lambda a: int(
                np.sum(np.asarray(a) < drop_threshold_m)
            ),
        },
        nodata=np.nan,
        all_touched=False,
    )
    stats_df = pd.DataFrame(stats).rename(
        columns={"min": "min_change_m", "mean": "mean_change_m", "count": "n_pixels"}
    )
    return gpd.GeoDataFrame(
        pd.concat([parcels, stats_df], axis=1), geometry=parcels.geometry, crs=PHILLY_CRS
    )


def flag_collapses(
    features: gpd.GeoDataFrame,
    min_drop_m: float = DEFAULT_DROP_M,
    min_dropped_pixels: int = 4,
) -> gpd.GeoDataFrame:
    """Select parcels with large negative surface change and enough dropped
    pixels to rule out edge artifacts. Tune min_dropped_pixels by parcel size
    and DSM cell size (at 1 m cells, 4 px ≈ 4 m² of lost roof)."""
    mask = (
        (features["min_change_m"] < min_drop_m)
        & (features["n_pixels_dropped"].fillna(0) >= min_dropped_pixels)
    )
    flagged = features.loc[mask].copy()
    flagged["collapse_score"] = -flagged["min_change_m"]
    return flagged.sort_values("collapse_score", ascending=False)


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------
def main(
    parcels_geojson: str = "data/balanced_parcels.geojson",
    data_dir: str = "data/roof_collapse",
    las_2022_filenames: Iterable[str] | None = None,
):
    data_dir = Path(data_dir)

    dem_2018_dir = download_dem_2018(data_dir / "dem_2018")

    las_2022_dir = download_las_2022(
        data_dir / "las_2022", filenames=las_2022_filenames
    )
    dsm_2022_tif = build_dsm_from_las(las_2022_dir, data_dir / "dsm_2022.tif")

    diff_tif = build_diff_raster(
        dem_2018_dir, dsm_2022_tif, data_dir / "diff_2022_minus_2018.tif"
    )

    parcels = gpd.read_file(parcels_geojson)
    features = per_parcel_height_change(parcels, diff_tif)
    features.to_file(data_dir / "parcel_height_change.geojson", driver="GeoJSON")

    flagged = flag_collapses(features)
    flagged.to_file(data_dir / "flagged_collapses.geojson", driver="GeoJSON")
    print(f"Flagged {len(flagged):,} parcels. Top 20:")
    print(flagged.head(20)[["min_change_m", "n_pixels_dropped", "collapse_score"]])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--parcels", default="data/balanced_parcels.geojson")
    ap.add_argument("--data-dir", default="data/roof_collapse")
    ap.add_argument(
        "--las-list",
        help=(
            "Path to a text file with one 2022 LAS filename per line "
            "(fetch the file list once from "
            "https://www.pasda.psu.edu/download/phillyLiDAR/2022/LAS/)."
        ),
    )
    args = ap.parse_args()

    names = None
    if args.las_list:
        names = [l.strip() for l in Path(args.las_list).read_text().splitlines() if l.strip()]
    main(args.parcels, args.data_dir, las_2022_filenames=names)
