"""
load_building_footprints.py
---------------------------
Fetches Philadelphia parcel boundaries live from OpenDataPhilly (ArcGIS).

Primary source : PWD Parcel layer (OpenDataPhilly / ArcGIS Hub GeoJSON)
                 https://opendata.arcgis.com/datasets/84baed491de44f539889f2af178ad85c_0.geojson

Responses are cached to disk so the API is only hit once per session.
Delete the cache file to force a fresh pull.

Returns a GeoDataFrame in EPSG:2272 (PA State Plane South, feet) with
all original PWD columns intact plus a normalised 'parcel_id' field.
"""

from pathlib import Path

import geopandas as gpd
import requests

# ── Configuration ─────────────────────────────────────────────────────────────
ODP_URLS = {
    "parcels": "https://opendata.arcgis.com/datasets/84baed491de44f539889f2af178ad85c_0.geojson",
}

TARGET_CRS = "EPSG:2272"   # PA State Plane South, feet
CACHE_DIR  = Path("data/vector")


def fetch_geojson(name: str, url: str, cache_dir: Path = CACHE_DIR) -> gpd.GeoDataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fpath = cache_dir / f"{name}.geojson"

    if fpath.exists():
        print(f"[load_building_footprints] Loading {name} from cache: {fpath}")
        return gpd.read_file(fpath)

    print(f"[load_building_footprints] Fetching {name} from {url} ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    fpath.write_bytes(r.content)
    print(f"  → cached to {fpath}")
    return gpd.read_file(fpath)


def load_building_footprints(
    url: str = ODP_URLS["parcels"],
    cache_dir: Path = CACHE_DIR,
    target_crs: str = TARGET_CRS,
) -> gpd.GeoDataFrame:
    parcels = fetch_geojson("parcels", url, cache_dir)
    parcels = parcels.to_crs(target_crs)

    # ── Geometry cleaning ──────────────────────────────────────────────────────
    before = len(parcels)

    # Drop null / empty geometries
    parcels = parcels[parcels.geometry.notna() & ~parcels.geometry.is_empty].copy()

    # Fix invalid geometries (self-intersections, bowtie polygons, etc.)
    from shapely.validation import make_valid
    parcels["geometry"] = parcels["geometry"].apply(make_valid)

    # Remove slivers smaller than 25 sq ft (noise / digitising artifacts)
    parcels = parcels[parcels.geometry.area > 25]

    # Filter out obvious non-building parcel types
    NON_BUILDING_TYPES = {"VACANT LAND", "PARKING LOT", "CEMETERY", "PARK", "WATER"}
    if "TYPEDESC" in parcels.columns:
        parcels = parcels[
            ~parcels["TYPEDESC"].str.upper().str.strip().isin(NON_BUILDING_TYPES)
        ]

    print(
        f"[load_building_footprints] Geometry cleaning: {before:,} → {len(parcels):,} parcels"
    )

    # Normalise parcel ID column to a consistent name
    pid_col = next(
        (c for c in parcels.columns
         if c.lower() in ("parcelid", "parcel_id", "objectid", "opa_account_num")),
        None,
    )
    if pid_col and pid_col != "parcel_id":
        parcels = parcels.rename(columns={pid_col: "parcel_id"})
    elif pid_col is None:
        parcels["parcel_id"] = parcels.index.astype(str)

    print(f"[load_building_footprints] Parcels loaded: {len(parcels):,} | CRS: {parcels.crs}")
    return parcels.reset_index(drop=True)


if __name__ == "__main__":
    gdf = load_building_footprints()
    print(gdf.head())
    print(f"Columns: {gdf.columns.tolist()}")