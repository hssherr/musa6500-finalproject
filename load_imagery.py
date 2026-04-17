"""
load_imagery.py
---------------
Loads the 2025 Philadelphia aerial imagery from a Cloud Optimized GeoTIFF
(COG) hosted on S3.  The file is opened lazily — no full download required.
GDAL streams only the tile windows that are actually read.

Usage
-----
    from load_imagery import open_imagery, read_parcel_chip

    src = open_imagery()            # lazy DataArray backed by the remote COG
    chip = read_parcel_chip(src, parcel_geom)   # clips one parcel window

If you need a local copy (e.g., for offline work), call download_cog() once
and then open_imagery(dest_path) to read from the local file.
"""

from pathlib import Path

import rioxarray as rxr
from shapely.geometry.base import BaseGeometry

COG_URL = "https://musa-650.s3.amazonaws.com/phl_aerial_0.3m.tif"
TARGET_CRS = "EPSG:2272"   # PA State Plane South, feet


def open_imagery(url: str = COG_URL):
    """
    Open the COG lazily via GDAL virtual filesystem.

    Parameters
    ----------
    url : str  Local path or remote https:// URL to the COG.

    Returns
    -------
    xarray.DataArray  Shape (bands, height, width).  Data is not loaded
    until a window is actually read.
    """
    src = rxr.open_rasterio(url, lock=False)
    print(f"[load_imagery] Opened: {url}")
    print(f"  CRS   : {src.rio.crs}")
    print(f"  Shape : {src.shape}  (bands × rows × cols)")
    print(f"  Res   : {src.rio.resolution()} (x_res, y_res)")
    return src


def read_parcel_chip(
    src,
    parcel_geom: BaseGeometry,
    crs: str = TARGET_CRS,
):
    """
    Clip the imagery to a single parcel geometry.

    Parameters
    ----------
    src         : DataArray returned by open_imagery()
    parcel_geom : shapely geometry in `crs` coordinates
    crs         : CRS of the input geometry

    Returns
    -------
    xarray.DataArray clipped to the parcel bounding box + mask.
    """
    return src.rio.clip([parcel_geom], crs=crs, from_disk=True)


def download_cog(
    url: str = COG_URL,
    dest: str = "data/imagery/phl_aerial_0.3m.tif",
) -> Path:
    """
    Download the full COG to disk.  Skips download if the file already exists.

    Parameters
    ----------
    url  : Source URL (default: S3 COG)
    dest : Local destination path

    Returns
    -------
    pathlib.Path  Path to the downloaded file.
    """
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"[load_imagery] Already downloaded: {dest}")
        return dest

    print(f"[load_imagery] Downloading {url} → {dest} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
    print(f"  → saved to {dest}")
    return dest


if __name__ == "__main__":
    src = open_imagery()
    print(src)