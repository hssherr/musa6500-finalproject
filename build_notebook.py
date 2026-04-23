"""Build main_v2.ipynb — a clean, end-to-end working version of the distress
detection notebook incorporating every fix from the code review."""

import json
from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


cells = []

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""# Detecting Structural Distress at Scale — v2
### A Geospatial Foundation Model Approach to Urban Building Safety

**Course:** MUSA 6500 — Geospatial Machine Learning in Remote Sensing
**Authors:** Jason Fan, Henry Sywulak-Herr

This is a clean rebuild of `main.ipynb` with the following fixes applied:

1. **40 m fixed-window chipping** (real imagery fills every chip instead of zero-padding around tight parcel bboxes)
2. **Clay NAIP RGB normalization** (frozen encoder was producing near-constant features on [0,1] input)
3. **Real lat/lon + time conditioning** (Clay was pretrained *with* these; all-zeros put the encoder off-distribution)
4. **Mean-pool over patch tokens** (CLS token in MAE-style models isn't discriminative)
5. **Full encoder unfreeze with LLRD** (4 layers at 1e-5 wasn't enough to bridge the Sentinel-2 → 0.3m aerial domain gap)
6. **Focal loss** (as the original markdown promised — actually implemented)
7. **Natural-distribution holdout set** (val on balanced data inflates precision)
8. **Early stopping + local checkpoint saves** (no more 20-epoch runs when plateaued; no more 1.2 GB writes to Drive mid-train)
9. **Feature diagnostic before training** (catches encoder collapse in 10 seconds, not 3 epochs)
10. **Chip cache on local SSD + one-shot Drive tarball** (avoids the 3-hour tar-from-Drive trap)
11. **Robust full-city inference** (on-the-fly chipping; previous setup silently failed on non-balanced parcels)
12. **Threshold tuning** on PR curve (default 0.5 is wrong for a non-50/50 prior)

Run top to bottom. The only manual toggle is `RUN_FULL_CITY` near the end."""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("## 1. Environment setup"))

cells.append(code("""import warnings
warnings.filterwarnings('ignore')

import os
import sys
import math
import shutil
import random
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f'  GPU: {torch.cuda.get_device_name(0)}')"""))

cells.append(code("""# ── Runtime setup (Colab vs local) ─────────────────────────────────────
import sys, os, subprocess

IS_COLAB = os.path.exists('/content')

if IS_COLAB:
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q',
        'git+https://github.com/Clay-foundation/model.git',
        'rioxarray', 'geopandas', 'tqdm',
    ], check=True)

    from google.colab import drive
    drive.mount('/content/drive')

    # Adjust this to match your Drive folder
    DRIVE_FOLDER = '/content/drive/MyDrive/folder'
    if DRIVE_FOLDER not in sys.path:
        sys.path.insert(0, DRIVE_FOLDER)

    DRIVE_BASE = Path(DRIVE_FOLDER)
    PROJECT_ROOT = Path('/content')
    print(f'Drive mounted. Project folder: {DRIVE_FOLDER}')
else:
    PROJECT_ROOT = Path('.').resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    DRIVE_BASE = PROJECT_ROOT
    print(f'Running locally. Project root: {PROJECT_ROOT}')

# Local SSD for chip cache — Drive is too slow for 23K small files.
# Only the tarball lives on Drive (fast single-file I/O).
CHIP_DIR = Path('/content/chips_local') if IS_COLAB else PROJECT_ROOT / 'data' / 'chips'
CHIP_DIR.mkdir(parents=True, exist_ok=True)

(DRIVE_BASE / 'data').mkdir(parents=True, exist_ok=True)
print(f'Chip cache: {CHIP_DIR}')"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 2. Data loading

Each dataset is loaded via its own dedicated module in the repo root."""))

cells.append(md("### Imagery (lazy COG from S3)"))

cells.append(code("""from load_imagery import open_imagery

# Lazy DataArray backed by a remote COG. GDAL streams only the tile
# windows we actually read.
src = open_imagery()"""))

cells.append(md("### Parcels and permits"))

cells.append(code("""from load_building_footprints import load_building_footprints, fetch_geojson

VECTOR_DIR = Path('data/vector')
CRS = 'EPSG:2272'

parcels = load_building_footprints()
print(f'Parcels: {len(parcels):,}')

PERMIT_URL = ('https://hub.arcgis.com/api/v3/datasets/'
              '8d18914ff740444793937d8724c64da8_0/downloads/data'
              '?format=geojson&spatialRefId=4326&where=1%3D1')
permits = fetch_geojson('permits', PERMIT_URL, VECTOR_DIR).to_crs(CRS)
print(f'Permits: {len(permits):,}')"""))

cells.append(md("### Labels (L&I violations)"))

cells.append(code("""from load_labels import load_labels, plot_labels

parcels_labeled = load_labels(parcels, permits=permits)

# Sanity check — flagged in code review as a silent-KeyError risk later
assert 'label' in parcels_labeled.columns, "Expected 'label' column"
assert 'label_permit_flagged' in parcels_labeled.columns, \\
    "load_labels.py output changed — 'label_permit_flagged' missing"
print("\\nLabel columns OK:",
      [c for c in parcels_labeled.columns if 'label' in c.lower()])

plot_labels(parcels_labeled)"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 3. Train / test split strategy

Two disjoint subsets:

1. **Natural-distribution holdout (`natural_test`)** — 10% of parcels at the
   real-world prior (~1.4% distressed). This is the number that goes in the
   final report — it's the only honest precision estimate.
2. **Balanced training set (`balanced_parcels`)** — all remaining distressed
   + 2× as many randomly sampled stable. Used for training and the
   balanced val set, which gives useful recall numbers but inflated precision."""))

cells.append(code("""NATURAL_TEST_FRAC = 0.10

rng = np.random.default_rng(SEED)
n = len(parcels_labeled)
natural_idx = rng.choice(n, size=int(n * NATURAL_TEST_FRAC), replace=False)
natural_mask = np.zeros(n, dtype=bool)
natural_mask[natural_idx] = True

natural_test = parcels_labeled[natural_mask].copy().reset_index(drop=True)
remaining    = parcels_labeled[~natural_mask].copy()

# Binary label on the natural set (0 vs any distress)
natural_test['binary_label'] = (natural_test['label'] > 0).astype(int)

print(f"Natural-distribution holdout: {len(natural_test):,}")
print(f"  Stable           : {(natural_test['binary_label']==0).sum():,}")
print(f"  Distressed       : {(natural_test['binary_label']==1).sum():,}")
print(f"  Real-world prior : {natural_test['binary_label'].mean():.4f}")"""))

cells.append(code("""# Balanced training set: 2:1 stable:distressed
distressed = remaining[remaining['label'].isin([1, 2])].copy()
num_stable = len(distressed) * 2

stable = (remaining[remaining['label'] == 0]
          .sample(n=num_stable, random_state=SEED).copy())

balanced_parcels = (pd.concat([distressed, stable])
                    .sample(frac=1, random_state=SEED)
                    .reset_index(drop=True))

# Merge Imminently Dangerous (2) into Unsafe (1) — binary task
balanced_parcels['label'] = balanced_parcels['label'].replace({2: 1})

print(f"Balanced training pool: {len(balanced_parcels):,}")
print(balanced_parcels['label'].value_counts())

# Persist for reproducibility
out = DRIVE_BASE / 'data' / 'balanced_parcels.geojson'
balanced_parcels[['geometry', 'label']].to_file(out, driver='GeoJSON')
print(f"Saved: {out}")

out_nat = DRIVE_BASE / 'data' / 'natural_test.geojson'
natural_test[['geometry', 'label', 'binary_label', 'label_permit_flagged']].to_file(
    out_nat, driver='GeoJSON'
)
print(f"Saved: {out_nat}")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("## 4. Clay checkpoint"))

cells.append(code("""import urllib.request

CKPT_PATH = DRIVE_BASE / "data" / "clay-v1.5.ckpt"
CKPT_URL  = "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"

if not CKPT_PATH.exists():
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Clay v1.5 checkpoint (~3 GB)…")
    urllib.request.urlretrieve(CKPT_URL, CKPT_PATH)
    print(f"Saved to {CKPT_PATH}")
else:
    size_mb = CKPT_PATH.stat().st_size / (1024 * 1024)
    print(f"Checkpoint already exists: {CKPT_PATH} ({size_mb:.0f} MB)")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 5. Chip cache

Instead of clipping tight to each parcel polygon and padding to square (which
made ~70% of every chip zero-padding), we sample a **fixed 130 ft (~40 m)
ground-area window centered on the parcel centroid**. Every chip is
end-to-end real imagery plus useful spatial context (neighboring buildings,
street, yard).

EPSG:2272 is in US survey feet. At 0.984 ft/px, 130 ft ≈ 132 pixels — the
dataset resizes to 224 for Clay.

### Cache strategy
- Chips live on local SSD (`/content/chips_local`) for fast I/O
- Only the tarball persists on Drive — a single 2 GB file transfers in
  seconds where 23K small files took hours"""))

cells.append(code("""from tqdm import tqdm

CHIP_FEET = 130.0  # ≈40 m at EPSG:2272 (which is US survey feet, not meters!)


def read_fixed_window(src, geom, size_feet: float = CHIP_FEET):
    \"\"\"Return a [3, H, W] float32 chip centered on geom's centroid.

    Units are CRS units (feet for EPSG:2272). 130 ft ≈ 40 m ≈ 130 px at 0.3m GSD.
    \"\"\"
    cx, cy = geom.centroid.x, geom.centroid.y
    half = size_feet / 2
    # rioxarray: y axis decreases in projected CRS (EPSG:2272)
    chip = src.sel(
        x=slice(cx - half, cx + half),
        y=slice(cy + half, cy - half),
    ).values
    if chip.ndim == 3 and chip.shape[0] >= 3:
        chip = chip[:3]
    else:
        raise ValueError(f"Unexpected chip shape: {chip.shape}")
    return chip.astype(np.float32)


def precache_chips(gdf, src, chip_dir, desc="Caching chips"):
    \"\"\"Clip and save every parcel chip. Skips already-saved chips.\"\"\"
    chip_dir.mkdir(parents=True, exist_ok=True)
    n_fallback = 0
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc=desc):
        out = chip_dir / f"{idx}.npy"
        if out.exists():
            continue
        try:
            chip = read_fixed_window(src, row.geometry)
            if chip.shape[1] < 10 or chip.shape[2] < 10:
                raise ValueError(f"Chip too small: {chip.shape}")
            np.save(out, chip)
        except Exception:
            np.save(out, np.zeros((3, 64, 64), dtype=np.float32))
            n_fallback += 1
    print(f"{desc} done. {n_fallback}/{len(gdf)} fallback chips.")"""))

cells.append(code("""# Ensure training chips are cached. Preference order:
#   1. Already present locally (no-op)
#   2. Extract from Drive tarball (fast)
#   3. Precache from S3 imagery (~20-30 min)

REBUILD_CHIPS = False  # set True to force re-chip (e.g., if you changed CHIP_FEET)

DRIVE_TAR = DRIVE_BASE / "data" / "chips.tar"
LOCAL_TAR = Path("/content/chips.tar") if IS_COLAB else PROJECT_ROOT / "data" / "chips.tar"

if REBUILD_CHIPS and CHIP_DIR.exists():
    print(f"REBUILD_CHIPS=True — wiping {CHIP_DIR}")
    shutil.rmtree(CHIP_DIR)
    CHIP_DIR.mkdir(parents=True, exist_ok=True)

existing = len(list(CHIP_DIR.glob('*.npy')))
needed = len(balanced_parcels)

if existing >= needed * 0.98:
    print(f"Chips already cached: {existing:,} files in {CHIP_DIR}")
elif DRIVE_TAR.exists() and not REBUILD_CHIPS:
    print(f"Restoring chips from {DRIVE_TAR}…")
    if IS_COLAB:
        os.system(f'cp "{DRIVE_TAR}" "{LOCAL_TAR}"')
        os.system(f'tar -xf "{LOCAL_TAR}" -C /content/')
    else:
        os.system(f'tar -xf "{DRIVE_TAR}" -C "{CHIP_DIR.parent}"')
    existing = len(list(CHIP_DIR.glob('*.npy')))
    print(f"Restored: {existing:,} chips")
    if existing < needed * 0.95:
        print(f"Tarball incomplete — precaching missing chips…")
        precache_chips(balanced_parcels, src, CHIP_DIR)
else:
    print(f"Precaching {needed:,} chips from S3 (expect ~20-30 min)…")
    precache_chips(balanced_parcels, src, CHIP_DIR)

# Build tarball for next session if we don't have one
if not DRIVE_TAR.exists() and IS_COLAB:
    print(f"Creating tarball for future sessions…")
    os.system(f'tar -cf "{LOCAL_TAR}" -C /content chips_local')
    os.system(f'cp "{LOCAL_TAR}" "{DRIVE_TAR}"')
    print(f"Saved tarball → {DRIVE_TAR}")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 6. Dataset and dataloaders

Key detail: chip normalization uses Clay's **NAIP RGB statistics**, the
closest match to Philadelphia 0.3 m aerial in Clay's pretraining set.
Without this the frozen encoder produces near-constant features (confirmed
empirically — `std_across_batch` went from 0.018 to >0.05 after adding
normalization)."""))

cells.append(code("""IMAGE_SIZE = 224  # Clay v1.5 expects 224x224

# Clay v1.5 NAIP RGB stats, from claymodel metadata.yaml
CLAY_RGB_MEAN = torch.tensor([110.16, 115.41,  98.15]).view(3, 1, 1) / 255.0
CLAY_RGB_STD  = torch.tensor([ 47.23,  39.82,  35.43]).view(3, 1, 1) / 255.0


def _pad_to_square_and_resize(arr: np.ndarray, size: int = IMAGE_SIZE) -> torch.Tensor:
    t = torch.from_numpy(arr)
    _, H, W = t.shape
    side = max(H, W)
    pad_h, pad_w = side - H, side - W
    t = F.pad(t, (pad_w // 2, pad_w - pad_w // 2,
                  pad_h // 2, pad_h - pad_h // 2), value=0.0)
    t = F.interpolate(t.unsqueeze(0), size=(size, size),
                      mode='bilinear', align_corners=False).squeeze(0)
    return t


class ParcelDataset(Dataset):
    \"\"\"Loads pre-cached chips, applies aug, normalizes to Clay stats.\"\"\"

    def __init__(self, gdf, imagery_src, transform=None, chip_dir=None,
                 label_col='label'):
        self.gdf = gdf
        self.src = imagery_src
        self.transform = transform
        self.chip_dir = Path(chip_dir) if chip_dir is not None else CHIP_DIR
        self.label_col = label_col

    def __len__(self):
        return len(self.gdf)

    def __getitem__(self, idx):
        row = self.gdf.iloc[idx]
        label = int(row[self.label_col])
        chip_id = row.name  # chip filename = df index

        chip_path = self.chip_dir / f"{chip_id}.npy"
        try:
            arr = np.load(chip_path).astype(np.float32)
            arr = np.clip(arr / 255.0, 0.0, 1.0)
        except Exception:
            arr = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        img = _pad_to_square_and_resize(arr, size=IMAGE_SIZE)

        if self.transform is not None:
            img = self.transform(img)

        # Normalize AFTER augmentation so ColorJitter etc operate on [0,1]
        img = (img - CLAY_RGB_MEAN) / CLAY_RGB_STD

        return img, torch.tensor(label, dtype=torch.long)"""))

cells.append(code("""import torchvision.transforms.v2 as Tv2

train_transforms = Tv2.Compose([
    Tv2.RandomHorizontalFlip(p=0.5),
    Tv2.RandomVerticalFlip(p=0.5),
    Tv2.RandomRotation(degrees=180),
    Tv2.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), antialias=True),
    Tv2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])

val_transforms = None"""))

cells.append(code("""from sklearn.model_selection import train_test_split

train_gdf, val_gdf = train_test_split(
    balanced_parcels,
    test_size=0.2,
    random_state=SEED,
    stratify=balanced_parcels['label'],
)

print(f"Training:   {len(train_gdf):,}")
print(f"Validation: {len(val_gdf):,}")

BATCH_SIZE = 64 if DEVICE == 'cuda' else 16
# num_workers=0: Colab's /dev/shm is too small + rioxarray src can't be
# pickled into worker subprocesses. Chips are local .npy so in-process is fine.
NUM_WORKERS = 0

train_loader = DataLoader(
    ParcelDataset(train_gdf, src, transform=train_transforms),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'),
)
val_loader = DataLoader(
    ParcelDataset(val_gdf, src, transform=val_transforms),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'),
)

# Smoke test — if this prints negative means near -1.8, chips are mostly
# zero-padding and something is wrong with the chip cache.
images, labels = next(iter(train_loader))
print(f"Batch: {tuple(images.shape)}  labels: {tuple(labels.shape)}")
print(f"Pixel stats (post-normalization):")
print(f"  min={images.min():.2f}  max={images.max():.2f}  mean={images.mean():.2f}")
print(f"  (mean near 0 = good; near -1.8 = chips are padding-dominated)")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 7. Model — Clay v1.5 + distress head

Changes vs. the v1 classifier:

- **Mean-pool patch tokens** instead of using `embeddings[:, 0, :]`.
  Clay is an MAE-style model — the [0] token is conditioning/register, not a
  learned CLS. Mean over patch tokens carries the actual visual content.
- **Real lat/lon and time conditioning.** Clay was pretrained with these as
  sinusoidal features; all-zeros puts the encoder off-distribution.
- **GSD = 1.0** (NAIP-ish) instead of 0.3. Clay has never seen 0.3 m imagery;
  pretending it's 1 m lands us inside the pretraining distribution.
- **Built-in 1-logit binary head** (no awkward replace-after-instantiate)."""))

cells.append(code("""import re
from claymodel.model import Encoder

# RGB wavelengths in micrometers (standard remote-sensing approximations)
RGB_WAVES = torch.tensor([0.665, 0.560, 0.493])

# GSD conditioning: 1.0m puts us inside Clay's training distribution
# (NAIP-like). Actual imagery is 0.3m but Clay has never seen that resolution.
RGB_GSD = torch.tensor(1.0)

# Philadelphia center for lat/lon conditioning
PHL_LAT = 39.9526
PHL_LON = -75.1652

# Approximate imagery acquisition window (Philly 2025 spring leaf-off flights)
IMAGERY_WEEK = 18   # early May
IMAGERY_HOUR = 11   # mid-morning


def _encode_latlon(lat: float, lon: float) -> torch.Tensor:
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    return torch.tensor([math.sin(lat_r), math.cos(lat_r),
                         math.sin(lon_r), math.cos(lon_r)], dtype=torch.float32)


def _encode_time(week: float, hour: float) -> torch.Tensor:
    w = 2 * math.pi * week / 52.0
    h = 2 * math.pi * hour / 24.0
    return torch.tensor([math.sin(w), math.cos(w),
                         math.sin(h), math.cos(h)], dtype=torch.float32)


PHL_LATLON   = _encode_latlon(PHL_LAT, PHL_LON)
IMAGERY_TIME = _encode_time(IMAGERY_WEEK, IMAGERY_HOUR)


class ClayDistressClassifier(nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()

        self.encoder = Encoder(
            mask_ratio=0.0, patch_size=8, shuffle=False,
            dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4.0,
        )

        # Binary head — 1 logit, sigmoid applied in criterion
        self.head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        if ckpt_path:
            self._load_clay_weights(ckpt_path)

    def _load_clay_weights(self, ckpt_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        encoder_sd = {
            re.sub(r"^model\\.encoder\\.", "", k): v
            for k, v in state_dict.items()
            if k.startswith("model.encoder")
        }
        missing = []
        for name, param in self.encoder.named_parameters():
            if name in encoder_sd and param.size() == encoder_sd[name].size():
                param.data.copy_(encoder_sd[name])
            else:
                missing.append(name)
        if missing:
            print(f"  [warn] {len(missing)} encoder params not loaded: {missing[:3]}…")
        else:
            print("  Clay encoder weights loaded.")

    def forward(self, pixels):
        B = pixels.shape[0]
        device = pixels.device
        datacube = {
            "pixels": pixels,
            "time":   IMAGERY_TIME.to(device).unsqueeze(0).expand(B, -1),
            "latlon": PHL_LATLON.to(device).unsqueeze(0).expand(B, -1),
            "gsd":    RGB_GSD.to(device),
            "waves":  RGB_WAVES.to(device),
        }
        embeddings, *_ = self.encoder(datacube)   # [B, 1+patches, 1024]
        # Mean-pool PATCH tokens only; [:, 0, :] is a conditioning token in Clay
        pooled = embeddings[:, 1:, :].mean(dim=1)  # [B, 1024]
        return self.head(pooled)


model = ClayDistressClassifier(ckpt_path=CKPT_PATH).to(DEVICE)

# Init head bias to logit(prior) so we don't waste epochs crawling out of 50/50
prior = float(train_gdf['label'].mean())
nn.init.constant_(model.head[-1].bias, math.log(prior / (1 - prior)))
print(f"\\nHead bias initialized: logit({prior:.3f}) = {math.log(prior/(1-prior)):.3f}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 8. Fine-tune setup — full encoder unfreeze with LLRD

We unfreeze the **entire encoder** (all 24 transformer layers) and use
**Layer-wise Learning Rate Decay**: the layer closest to the head gets
`BASE_LR`; each earlier layer gets `DECAY × ` the next layer's LR.

This lets the deepest, most task-specific layers adapt aggressively while
early layers (which encode generic low-level features) stay close to
pretrained. 4-layer unfreeze at 1e-5 wasn't enough to bridge the Sentinel-2
→ aerial domain gap; confirmed empirically that loss stayed at the prior
entropy."""))

cells.append(code("""for p in model.encoder.parameters():
    p.requires_grad = True

NUM_ENCODER_LAYERS = len(model.encoder.transformer.layers)
BASE_LR = 3e-5   # deepest encoder layer
DECAY   = 0.80   # each earlier layer ×0.8

param_groups = []

for i, layer in enumerate(model.encoder.transformer.layers):
    # Layer 23 (last, closest to head) gets BASE_LR
    lr_i = BASE_LR * (DECAY ** (NUM_ENCODER_LAYERS - 1 - i))
    param_groups.append({"params": layer.parameters(), "lr": lr_i})

# patch_embedding + any other non-transformer encoder bits
other = [p for n, p in model.encoder.named_parameters()
         if not n.startswith("transformer.layers") and p.requires_grad]
param_groups.append({
    "params": other,
    "lr": BASE_LR * (DECAY ** NUM_ENCODER_LAYERS),
})

param_groups.append({"params": model.head.parameters(), "lr": 5e-4})

optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

lr_earliest = BASE_LR * (DECAY ** (NUM_ENCODER_LAYERS - 1))
print(f"LLRD schedule:")
print(f"  head        : 5.0e-04")
print(f"  encoder[23] : {BASE_LR:.1e}   (closest to head)")
print(f"  encoder[00] : {lr_earliest:.1e}   (furthest from head)")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 9. Feature diagnostic — run BEFORE training

A 10-second check that the encoder is producing **discriminative** features.
If `std_across_batch < 0.03`, the encoder is outputting near-constant vectors
regardless of input — no head will save you, and training will plateau at
the prior's entropy."""))

cells.append(code("""model.eval()
with torch.no_grad():
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(DEVICE)
    B = imgs.shape[0]
    datacube = {
        "pixels": imgs,
        "time":   IMAGERY_TIME.to(DEVICE).unsqueeze(0).expand(B, -1),
        "latlon": PHL_LATLON.to(DEVICE).unsqueeze(0).expand(B, -1),
        "gsd":    RGB_GSD.to(DEVICE),
        "waves":  RGB_WAVES.to(DEVICE),
    }
    embeddings, *_ = model.encoder(datacube)
    pooled = embeddings[:, 1:, :].mean(dim=1)

mean_val = pooled.mean().item()
std_within = pooled.std(dim=1).mean().item()
std_across = pooled.std(dim=0).mean().item()

print(f"Pre-training feature check:")
print(f"  mean              = {mean_val:+.3f}")
print(f"  std_within_sample = {std_within:.3f}")
print(f"  std_across_batch  = {std_across:.4f}")

if std_across < 0.03:
    print("\\n⚠️  std_across_batch very low — encoder is not responding to inputs.")
    print("   Check: chip cache, Clay normalization, RGB_GSD, lat/lon conditioning.")
elif std_across < 0.05:
    print("\\n⚠️  Borderline — training may work but slowly. Consider increasing BASE_LR.")
else:
    print("\\n✓  Encoder is discriminating inputs — safe to train.")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 10. Focal loss

Binary focal loss with `γ=2.0`. Downweights "easy" examples (confidently
correct) so the model can't coast on the abundant clearly-stable buildings.

`α=0.5` is correct here because the training pool is already down-sampled
to 2:1 — we don't want to double-count the minority class."""))

cells.append(code("""class FocalLoss(nn.Module):
    \"\"\"Binary focal loss. Reduces to BCE at gamma=0, alpha=0.5.\"\"\"

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)   # p assigned to the correct class
        focal = (1 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal = alpha_t * focal
        return focal.mean()


criterion = FocalLoss(gamma=2.0, alpha=0.5)"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 11. Training loop

Features:
- **OneCycleLR** with 10% warmup over all param groups (each group uses its
  own `max_lr` from the LLRD schedule)
- **AMP (fp16)** for ~2× A100 speedup
- **Early stopping** with `PATIENCE=4`
- **Best-checkpoint gate**: only saves when `pos_rate` is in a sane range
  (avoids saving models that collapsed to all-positive)
- **Local save** to `/content/best_model.pt` — copied to Drive once at end
  (previous setup wrote 1.2 GB to Drive every "improved" epoch)
- **LR logging** each epoch so you can debug OneCycle's warmup/peak/decay"""))

cells.append(code("""from sklearn.metrics import recall_score, precision_score, classification_report

NUM_EPOCHS  = 20
THRESHOLD   = 0.33      # matches the balanced prior (1/3)
PATIENCE    = 4
USE_AMP     = True

best_recall = 0.0
best_epoch  = 0
since_improvement = 0

LOCAL_BEST = Path("/content/best_model.pt") if IS_COLAB else PROJECT_ROOT / "data" / "best_model.pt"

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[g["lr"] for g in optimizer.param_groups],
    epochs=NUM_EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
)

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

print(f"Training up to {NUM_EPOCHS} epochs, patience={PATIENCE}, threshold={THRESHOLD}")
print(f"Saving best to: {LOCAL_BEST}")

for epoch in range(NUM_EPOCHS):
    # ── Train ────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, labels)

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        train_loss += loss.item()

    # ── Validate ─────────────────────────────────────────────────────
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                probs = torch.sigmoid(model(images)).squeeze(1).float()
            preds = (probs > THRESHOLD).long()
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    recall    = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    pos_rate  = sum(all_preds) / len(all_preds)

    # param_groups order: layer_0..layer_23, other_encoder, head
    cur_head_lr    = optimizer.param_groups[-1]['lr']
    cur_encoder_lr = optimizer.param_groups[-3]['lr']  # layer 23 (deepest, highest LR)

    print(f"Ep {epoch+1:02d} | loss {train_loss/len(train_loader):.4f} "
          f"| R {recall:.3f} | P {precision:.3f} | pos {pos_rate:.2f} "
          f"| lr_head {cur_head_lr:.1e} lr_enc23 {cur_encoder_lr:.1e}")

    improved = (recall > best_recall) and (0.10 < pos_rate < 0.75)
    if improved:
        best_recall = recall
        best_epoch = epoch + 1
        since_improvement = 0
        torch.save(model.state_dict(), LOCAL_BEST)
        print(f"  → saved (best R={best_recall:.3f} @ ep {best_epoch})")
    else:
        since_improvement += 1

    if since_improvement >= PATIENCE:
        print(f"\\nEarly stopping: no improvement for {PATIENCE} epochs.")
        break

print(f"\\nBest recall: {best_recall:.3f} at epoch {best_epoch}")"""))

cells.append(code("""# Copy best checkpoint to Drive once, after training
DRIVE_BEST = DRIVE_BASE / "data" / "best_model.pt"
if LOCAL_BEST.exists():
    shutil.copy(LOCAL_BEST, DRIVE_BEST)
    size_mb = LOCAL_BEST.stat().st_size / (1024 * 1024)
    print(f"Copied best model ({size_mb:.0f} MB) → {DRIVE_BEST}")
else:
    print("⚠️  No best_model.pt — training never found a non-collapsed epoch.")
    print("   Check feature diagnostic above and loss trajectory.")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 12. Evaluation on balanced validation set"""))

cells.append(code("""from sklearn.metrics import ConfusionMatrixDisplay, classification_report

if not LOCAL_BEST.exists():
    raise RuntimeError(
        f"No checkpoint at {LOCAL_BEST}. Training never found an epoch with "
        "recall > 0 in the valid pos_rate range. Check the feature diagnostic "
        "output and loss trajectory — likely an encoder collapse issue."
    )

model.load_state_dict(torch.load(LOCAL_BEST, map_location=DEVICE))
model.eval()

all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        probs = torch.sigmoid(model(images)).squeeze(1)
        preds = (probs > THRESHOLD).long()
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

print(f"Validation report (balanced set, threshold={THRESHOLD}):")
print(classification_report(all_labels, all_preds,
                            target_names=["Stable", "Distressed"], zero_division=0))

ConfusionMatrixDisplay.from_predictions(
    all_labels, all_preds,
    display_labels=["Stable", "Distressed"], cmap="Blues",
)
plt.title("Validation Confusion Matrix (balanced set)")
plt.tight_layout()
plt.show()"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 13. Threshold tuning

Pick the operating point that maximizes F1 on the validation PR curve.
If missing a distressed building is worse than a false alarm, bias toward
higher recall (lower threshold)."""))

cells.append(code("""from sklearn.metrics import precision_recall_curve

probs  = np.asarray(all_probs)
labels_np = np.asarray(all_labels)

precisions, recalls, thresholds = precision_recall_curve(labels_np, probs)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = f1s[:-1].argmax()
THRESHOLD_TUNED = float(thresholds[best_idx])

print(f"Best-F1 threshold: {THRESHOLD_TUNED:.3f}")
print(f"  P={precisions[best_idx]:.3f}  R={recalls[best_idx]:.3f}  F1={f1s[best_idx]:.3f}\\n")

print(f"{'thr':>5}  {'recall':>7}  {'prec':>7}  {'f1':>6}  {'pos_rate':>8}")
for thr in [0.20, 0.25, 0.30, 0.33, 0.40, 0.50, 0.60, 0.70]:
    p = (probs > thr).astype(int)
    tp = int(((p == 1) & (labels_np == 1)).sum())
    fp = int(((p == 1) & (labels_np == 0)).sum())
    fn = int(((p == 0) & (labels_np == 1)).sum())
    rec  = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"{thr:5.2f}  {rec:7.3f}  {prec:7.3f}  {f1:6.3f}  {p.mean():8.2f}")

print(f"\\n--- Classification at tuned threshold {THRESHOLD_TUNED:.3f} ---")
print(classification_report(
    labels_np, (probs > THRESHOLD_TUNED).astype(int),
    target_names=["Stable", "Distressed"], zero_division=0,
))"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 14. Natural-distribution holdout — the honest number

The balanced val set gives inflated precision because its prior is 33%.
The real-world prior is ~1.4%. This cell re-evaluates on a sample at the
natural distribution so you can report defensible metrics in the final writeup."""))

cells.append(code("""# Chip the natural-distribution holdout (separate cache dir)
natural_chip_dir = Path("/content/natural_chips") if IS_COLAB else PROJECT_ROOT / "data" / "natural_chips"
natural_chip_dir.mkdir(parents=True, exist_ok=True)

print(f"Chipping {len(natural_test):,} natural-distribution parcels…")
precache_chips(natural_test, src, natural_chip_dir, desc="Natural chips")

nat_loader = DataLoader(
    ParcelDataset(natural_test, src, transform=None,
                  chip_dir=natural_chip_dir, label_col='binary_label'),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'),
)

model.eval()
nat_probs, nat_preds, nat_labels = [], [], []
with torch.no_grad():
    for images, labels in nat_loader:
        images = images.to(DEVICE)
        probs = torch.sigmoid(model(images)).squeeze(1)
        preds = (probs > THRESHOLD_TUNED).long()
        nat_probs.extend(probs.cpu().tolist())
        nat_preds.extend(preds.cpu().tolist())
        nat_labels.extend(labels.tolist())

print(f"\\nNatural-distribution test set (threshold={THRESHOLD_TUNED:.3f}):")
print(f"  n={len(nat_labels):,}  true positives prior={np.mean(nat_labels):.4f}")
print(f"  predicted positive rate: {np.mean(nat_preds):.4f}")
print(classification_report(nat_labels, nat_preds,
                            target_names=["Stable", "Distressed"], zero_division=0))

ConfusionMatrixDisplay.from_predictions(
    nat_labels, nat_preds,
    display_labels=["Stable", "Distressed"], cmap="Oranges",
)
plt.title("Natural-Distribution Confusion Matrix")
plt.tight_layout()
plt.show()"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 15. Geographic stratification

Check whether model performance varies by region. North/South/West Philly
have different building stocks and distress patterns."""))

cells.append(code("""HOOD_URL = ("https://raw.githubusercontent.com/azavea/geo-data/master/"
            "Neighborhoods_Philadelphia/Neighborhoods_Philadelphia.geojson")
hoods = gpd.read_file(HOOD_URL).to_crs("EPSG:2272")
print(f"Loaded {len(hoods)} neighborhoods")

REGION_MAP = {
    "NORTH": [
        "Kensington", "Frankford", "Olney", "Logan", "Hunting Park",
        "Fairhill", "Feltonville", "Juniata Park", "Strawberry Mansion",
        "North Central", "Brewerytown", "Fishtown",
    ],
    "SOUTH": [
        "South Philadelphia", "Passyunk Square", "Point Breeze",
        "Pennsport", "Grays Ferry", "Whitman", "Girard Estates",
    ],
    "WEST": [
        "West Philadelphia", "Cobbs Creek", "Overbrook", "Mantua",
        "Mill Creek", "Powelton Village", "Belmont", "Haddington",
    ],
}

# Use the balanced val set (has enough positives per region for meaningful recall).
# Re-derive preds at the tuned threshold so numbers match the overall eval.
val_with_preds = val_gdf.copy().reset_index(drop=True)
val_with_preds["pred"]  = (np.asarray(all_probs) > THRESHOLD_TUNED).astype(int)
val_with_preds["label"] = all_labels
val_with_hood = gpd.sjoin(val_with_preds, hoods[["geometry", "name"]], how="left")

for region, names in REGION_MAP.items():
    sub = val_with_hood[val_with_hood["name"].isin(names)]
    if len(sub) == 0:
        print(f"{region:6s} | no parcels matched")
        continue
    r = recall_score(sub["label"], sub["pred"], pos_label=1, zero_division=0)
    p = precision_score(sub["label"], sub["pred"], pos_label=1, zero_division=0)
    print(f"{region:6s} | n={len(sub):4d} | recall={r:.3f} | precision={p:.3f}")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 16. Full-city inference (OPTIONAL — 30+ minutes)

Gated behind `RUN_FULL_CITY = False`. Flip to `True` to generate predictions
for every parcel in the city, apply the permit filter, and export a GeoJSON.

This precaches ~547K chips (~30 min first run, then cached) and runs
inference over all of them. Previous version of the notebook silently
failed here because chips were only cached for `balanced_parcels`."""))

cells.append(code("""RUN_FULL_CITY = False

if not RUN_FULL_CITY:
    print("Skipping full-city inference. Set RUN_FULL_CITY=True to run.")
else:
    full_chip_dir = Path("/content/full_chips") if IS_COLAB else PROJECT_ROOT / "data" / "full_chips"
    full_chip_dir.mkdir(parents=True, exist_ok=True)

    # Work on a fresh copy with a clean integer index for chip filenames
    full_df = parcels_labeled.copy().reset_index(drop=True)
    full_df['binary_label'] = (full_df['label'] > 0).astype(int)

    print(f"Chipping {len(full_df):,} parcels (≈30 min first run)…")
    precache_chips(full_df, src, full_chip_dir, desc="Full-city chips")

    full_loader = DataLoader(
        ParcelDataset(full_df, src, transform=None,
                      chip_dir=full_chip_dir, label_col='binary_label'),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'),
    )

    model.eval()
    all_city_probs = []
    with torch.no_grad():
        for i, (images, _) in enumerate(full_loader):
            images = images.to(DEVICE)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                probs = torch.sigmoid(model(images)).squeeze(1).float()
            all_city_probs.extend(probs.cpu().tolist())
            if i % 200 == 0:
                print(f"  {i * BATCH_SIZE:,} / {len(full_df):,} parcels scored")

    full_df["distress_score"]  = all_city_probs
    full_df["pred_distressed"] = (full_df["distress_score"] > THRESHOLD_TUNED).astype(int)

    # Permit filter — zero out predictions for parcels with active eCLIPSE permits
    full_df.loc[full_df["label_permit_flagged"], "pred_distressed"] = 0

    print("\\nPrediction counts after permit filter:")
    print(full_df["pred_distressed"].value_counts())

    out_path = DRIVE_BASE / "output" / "predictions.geojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["geometry", "parcel_id", "distress_score", "pred_distressed", "label"]
    full_df[cols].to_file(out_path, driver="GeoJSON")
    print(f"\\nSaved: {out_path}")"""))

cells.append(code("""if RUN_FULL_CITY:
    fig, ax = plt.subplots(figsize=(10, 10))
    full_df[full_df["pred_distressed"] == 0].plot(
        ax=ax, color="steelblue", alpha=0.2, linewidth=0,
    )
    full_df[full_df["pred_distressed"] == 1].plot(
        ax=ax, column="distress_score", cmap="YlOrRd", linewidth=0, legend=True,
    )
    ax.set_title("Predicted Structurally Distressed Parcels — Philadelphia")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
else:
    print("Map skipped (RUN_FULL_CITY=False)")"""))

# ─────────────────────────────────────────────────────────────────────────

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10",
    },
}

out_path = Path("F:/GitHub/musa6500-finalproject/main_v2.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path}")
print(f"Cells: {len(cells)} "
      f"({sum(1 for c in cells if c.cell_type == 'code')} code, "
      f"{sum(1 for c in cells if c.cell_type == 'markdown')} markdown)")
