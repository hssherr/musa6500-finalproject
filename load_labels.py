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
    1 = Unsafe / Clean & Seal
    2 = Imminently Dangerous   (overwrites Unsafe if a parcel has both)

A boolean 'label_permit_flagged' column is added for label=2 parcels that
also have an active eCLIPSE permit — likely new construction, not decay.

Responses are cached to disk so the API is only hit once per session.
Delete the cache files to force a fresh pull.
"""

import io
import urllib.parse
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import requests

CARTO_BASE = "https://phl.carto.com/api/v2/sql"
TARGET_CRS = "EPSG:2272"
CACHE_DIR  = Path("data/labels")

# Only fetch violations whose case is still open and created after this date.
CUTOFF_DATE = "2020-01-01"

_WHERE: dict[str, str] = {
    "imm_dang":   f"casecreateddate > '{CUTOFF_DATE}'",
    "unsafe":     f"casecreateddate > '{CUTOFF_DATE}'",
    "clean_seal": f"casecreateddate > '{CUTOFF_DATE}'",
}


def fetch_carto_geojson(table: str, cache_dir: Path = CACHE_DIR) -> gpd.GeoDataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fpath = cache_dir / f"{table}.geojson"

    if fpath.exists():
        print(f"[load_labels] Loading {table} from cache: {fpath}")
        return gpd.read_file(fpath)

    where = _WHERE.get(table, "")
    query = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "")
    
    print(f"[load_labels] Fetching {table} from Carto API...")
    r = requests.get(
        CARTO_BASE, 
        params={"format": "GeoJSON", "q": query}, 
        timeout=120
    )
    
    # Check if Carto rejected the SQL query before trying to save it
    if not r.ok:
        print(f"\n[!] CARTO API ERROR on table '{table}':")
        print(f"Query sent: {query}")
        print(f"Error msg : {r.text}\n")
        r.raise_for_status()

    fpath.write_bytes(r.content)
    print(f"  → cached to {fpath}")
    return gpd.read_file(io.BytesIO(r.content), driver="GeoJSON")


def _sjoin_predicate(violations: gpd.GeoDataFrame) -> str:
    """Return 'within' for point layers, 'intersects' for polygon layers."""
    if violations.geometry.geom_type.isin(["Point", "MultiPoint"]).all():
        return "within"
    return "intersects"


def load_labels(
    parcels: gpd.GeoDataFrame,
    permits: gpd.GeoDataFrame = None,
    cache_dir: Path = CACHE_DIR,
) -> gpd.GeoDataFrame:
    parcels = parcels.copy()

    print("Fetching imm_dang...")
    imm_dang = fetch_carto_geojson("imm_dang", cache_dir).to_crs(parcels.crs)
    print(f"  rows: {len(imm_dang):,}")

    print("Fetching unsafe...")
    unsafe = fetch_carto_geojson("unsafe", cache_dir).to_crs(parcels.crs)
    print(f"  rows: {len(unsafe):,}")

    print("Fetching clean_seal...")
    clean_seal = fetch_carto_geojson("clean_seal", cache_dir).to_crs(parcels.crs)
    print(f"  rows: {len(clean_seal):,}")

    id_parcels = gpd.sjoin(
        parcels[["geometry"]],
        imm_dang[["geometry"]],
        how="inner",
        predicate=_sjoin_predicate(imm_dang),
    ).index.unique()

    unsafe_parcels = gpd.sjoin(
        parcels[["geometry"]],
        unsafe[["geometry"]],
        how="inner",
        predicate=_sjoin_predicate(unsafe),
    ).index.unique()

    cs_parcels = gpd.sjoin(
        parcels[["geometry"]],
        clean_seal[["geometry"]],
        how="inner",
        predicate=_sjoin_predicate(clean_seal),
    ).index.unique()

    parcels["label"] = 0
    parcels.loc[parcels.index.isin(cs_parcels),     "label"] = 1
    parcels.loc[parcels.index.isin(unsafe_parcels), "label"] = 1
    parcels.loc[parcels.index.isin(id_parcels),     "label"] = 2

    print(f"Labeled Imminently Dangerous : {(parcels['label'] == 2).sum():,}")
    print(f"Labeled Unsafe               : {(parcels['label'] == 1).sum():,}")
    print(f"Labeled Stable               : {(parcels['label'] == 0).sum():,}")

    # Flag label=2 parcels that also have an active eCLIPSE permit
    parcels["label_permit_flagged"] = False
    if permits is not None and "status" in permits.columns:
        active_permit_idx = gpd.sjoin(
            parcels.loc[parcels["label"] == 2, ["geometry"]],
            permits.loc[permits["status"] == "ISSUED", ["geometry"]].to_crs(parcels.crs),
            how="inner",
            predicate="intersects",
        ).index.unique()
        parcels.loc[
            parcels.index.isin(active_permit_idx) & (parcels["label"] == 2),
            "label_permit_flagged",
        ] = True
        print(f"Permit-flagged ID parcels    : {parcels['label_permit_flagged'].sum():,}")

    return parcels


def plot_labels(parcels: gpd.GeoDataFrame) -> None:
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


if __name__ == "__main__":
    from load_building_footprints import load_building_footprints
    parcels = load_building_footprints()
    labeled = load_labels(parcels)
    print(labeled["label"].value_counts())
    plot_labels(labeled)