"""
load_labels.py
--------------
Fetches L&I violation data live from the Philadelphia Carto API and
produces multi-class training labels via spatial join against parcels.

Sources (Carto API):
    imm_dang   – Imminently Dangerous structures
    unsafe     – Unsafe structures
    clean_seal – Clean & Seal orders

Labels:
    0 = Stable
    1 = Unsafe
    2 = Imminently Dangerous   (overwrites Unsafe if a parcel has both)

Responses are cached to disk so the API is only hit once per session.
Delete the cache files to force a fresh pull.
"""

import io
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import requests

# ── Configuration ─────────────────────────────────────────────────────────────
CARTO_BASE = "https://phl.carto.com/api/v2/sql"

TARGET_CRS = "EPSG:2272"   # PA State Plane South, feet
CACHE_DIR  = Path("data/labels")


# ── Core fetch helper (mirrors your fetch_carto_geojson pattern) ──────────────
def fetch_carto_geojson(table: str, cache_dir: Path = CACHE_DIR) -> gpd.GeoDataFrame:
    """
    Pull a full table from the Philadelphia Carto API as GeoJSON.
    Caches result to <cache_dir>/<table>.geojson to avoid repeat downloads.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fpath = cache_dir / f"{table}.geojson"

    if fpath.exists():
        print(f"[load_labels] Loading {table} from cache: {fpath}")
        return gpd.read_file(fpath)

    url = f"{CARTO_BASE}?format=GeoJSON&q=SELECT * FROM {table}"
    print(f"[load_labels] Fetching {table} from Carto API...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    fpath.write_bytes(r.content)
    print(f"  → cached to {fpath}")
    return gpd.read_file(io.BytesIO(r.content), driver="GeoJSON")


# ── Public API ────────────────────────────────────────────────────────────────
def load_labels(parcels: gpd.GeoDataFrame, cache_dir: Path = CACHE_DIR) -> gpd.GeoDataFrame:
    """
    Fetch violation tables, spatially join to parcels, and assign labels.

    Parameters
    ----------
    parcels   : GeoDataFrame of parcel polygons (must already be in TARGET_CRS)
    cache_dir : directory for cached GeoJSON files

    Returns
    -------
    parcels GeoDataFrame with an added 'label' column:
        0 = Stable
        1 = Unsafe
        2 = Imminently Dangerous
    """
    parcels = parcels.copy()

    # ── Fetch violation layers ─────────────────────────────────────────────────
    print("Fetching imm_dang...")
    imm_dang = fetch_carto_geojson("imm_dang", cache_dir).to_crs(parcels.crs)
    print(f"  rows: {len(imm_dang):,}")

    print("Fetching unsafe...")
    unsafe = fetch_carto_geojson("unsafe", cache_dir).to_crs(parcels.crs)
    print(f"  rows: {len(unsafe):,}")

    print("Fetching clean_seal...")
    clean_seal = fetch_carto_geojson("clean_seal", cache_dir).to_crs(parcels.crs)
    print(f"  rows: {len(clean_seal):,}")

    # ── Spatial joins ──────────────────────────────────────────────────────────
    id_parcels = gpd.sjoin(
        parcels[["geometry"]],
        imm_dang[["geometry"]],
        how="inner",
        predicate="intersects",
    ).index.unique()

    unsafe_parcels = gpd.sjoin(
        parcels[["geometry"]],
        unsafe[["geometry"]],
        how="inner",
        predicate="intersects",
    ).index.unique()

    cs_parcels = gpd.sjoin(
        parcels[["geometry"]],
        clean_seal[["geometry"]],
        how="inner",
        predicate="intersects",
    ).index.unique()

    # ── Assign labels (ID overwrites Unsafe if a parcel has both) ─────────────
    parcels["label"] = 0
    parcels.loc[parcels.index.isin(cs_parcels),     "label"] = 1
    parcels.loc[parcels.index.isin(unsafe_parcels), "label"] = 1
    parcels.loc[parcels.index.isin(id_parcels),     "label"] = 2

    print(f"Labeled Imminently Dangerous : {(parcels['label'] == 2).sum():,}")
    print(f"Labeled Unsafe               : {(parcels['label'] == 1).sum():,}")
    print(f"Labeled Stable               : {(parcels['label'] == 0).sum():,}")

    return parcels


def plot_labels(parcels: gpd.GeoDataFrame) -> None:
    """Quick visualisation of the three label classes."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    parcels[parcels.label == 0].plot(ax=ax, color="steelblue", alpha=0.3, linewidth=0)
    parcels[parcels.label == 1].plot(ax=ax, color="orange",    alpha=0.8, linewidth=0)
    parcels[parcels.label == 2].plot(ax=ax, color="crimson",   alpha=0.8, linewidth=0)

    st_patch = mpatches.Patch(color="steelblue", label=f"Stable ({(parcels['label'] == 0).sum():,})")
    us_patch = mpatches.Patch(color="orange",    label=f"Unsafe ({(parcels['label'] == 1).sum():,})")
    id_patch = mpatches.Patch(color="crimson",   label=f"Imminently Dangerous ({(parcels['label'] == 2).sum():,})")

    ax.legend(handles=[id_patch, us_patch, st_patch])
    ax.set_title("Labeled Parcels — Imminently Dangerous, Unsafe & Stable")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from load_building_footprints import load_building_footprints
    parcels = load_building_footprints()
    labeled = load_labels(parcels)
    print(labeled["label"].value_counts())
    plot_labels(labeled)
