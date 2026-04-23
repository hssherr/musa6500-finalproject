"""main_v4.ipynb — Clay embeddings + RandomForest classifier.

This follows Clay's OFFICIAL classification recipe from
docs/finetune/finetune-on-embeddings.ipynb:

  1. Extract Clay embeddings once per chip, cache to disk
  2. Train sklearn RandomForestClassifier on embeddings
  3. Evaluate

No training loops, no LLRD, no AMP. Pure feature-extraction + classical ML.
If this doesn't work, nothing in Clay's ecosystem will.

Key differences from main_v2.ipynb:
  - IMAGE_SIZE = 256 (Clay's native resolution, gives 32×32 = 1024 patches)
  - ClayMAEModule.load_from_checkpoint gets explicit model_size/dolls/doll_weights
  - Extract embeddings once, then classical ML on top
  - Prediction via predict_proba for threshold tuning
"""

from pathlib import Path
import nbformat as nbf


def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)


cells = []

cells.append(md("""# Detecting Structural Distress at Scale — v4 (Clay Embeddings + RandomForest)

**Course:** MUSA 6500 — Geospatial Machine Learning in Remote Sensing
**Authors:** Jason Fan, Henry Sywulak-Herr

## Why this approach

Clay's official downstream-classification recipe
([docs/finetune/finetune-on-embeddings.ipynb](https://clay-foundation.github.io/model/finetune/finetune-on-embeddings.html))
is **NOT neural-network fine-tuning**. It's:

1. Run Clay as a pure feature extractor, cache embeddings per chip
2. Train `sklearn.ensemble.RandomForestClassifier` on those cached embeddings

The tutorial achieves 90% accuracy on 216 marina-detection samples. No training loops,
no AMP, no gradient-based fine-tuning at all.

Previous attempts (main_v2.ipynb) used LLRD and neural-network heads, which is
outside Clay's documented use pattern for classification. This notebook follows the
official pattern exactly.

## Key Clay-specific settings

- **IMAGE_SIZE = 256** (Clay's native resolution → 32×32 = 1024 patch tokens)
- **ClayMAEModule.load_from_checkpoint** with explicit `model_size`, `dolls`, `doll_weights`
  as prescribed by the embeddings tutorial
- **LINZ-aligned conditioning** (closest pretraining platform to 0.3 m aerial RGB)
- **Zero time/latlon** (sanctioned by Clay docs: *"can be set to zero if not available"*)
- **CLS token readout** (matches claymodel/finetune/classify/factory.py)"""))

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

cells.append(code("""import sys, os, subprocess

IS_COLAB = os.path.exists('/content')

if IS_COLAB:
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q',
        'git+https://github.com/Clay-foundation/model.git',
        'rioxarray', 'geopandas', 'tqdm', 'scikit-learn',
    ], check=True)

    from google.colab import drive
    drive.mount('/content/drive')

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

CHIP_DIR = Path('/content/chips_local') if IS_COLAB else PROJECT_ROOT / 'data' / 'chips'
CHIP_DIR.mkdir(parents=True, exist_ok=True)
(DRIVE_BASE / 'data').mkdir(parents=True, exist_ok=True)
print(f'Chip cache: {CHIP_DIR}')"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("## 2. Data loading"))

cells.append(code("""from load_imagery import open_imagery

src = open_imagery()"""))

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

cells.append(code("""from load_labels import load_labels, plot_labels

parcels_labeled = load_labels(parcels, permits=permits)

assert 'label' in parcels_labeled.columns
assert 'label_permit_flagged' in parcels_labeled.columns
print("\\nLabel columns OK:",
      [c for c in parcels_labeled.columns if 'label' in c.lower()])

plot_labels(parcels_labeled)"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 3. Train / test split strategy"""))

cells.append(code("""NATURAL_TEST_FRAC = 0.10

rng = np.random.default_rng(SEED)
n = len(parcels_labeled)
natural_idx = rng.choice(n, size=int(n * NATURAL_TEST_FRAC), replace=False)
natural_mask = np.zeros(n, dtype=bool)
natural_mask[natural_idx] = True

natural_test = parcels_labeled[natural_mask].copy().reset_index(drop=True)
remaining    = parcels_labeled[~natural_mask].copy()

natural_test['binary_label'] = (natural_test['label'] > 0).astype(int)

print(f"Natural-distribution holdout: {len(natural_test):,}")
print(f"  Stable     : {(natural_test['binary_label']==0).sum():,}")
print(f"  Distressed : {(natural_test['binary_label']==1).sum():,}")
print(f"  Real prior : {natural_test['binary_label'].mean():.4f}")"""))

cells.append(code("""distressed = remaining[remaining['label'].isin([1, 2])].copy()
num_stable = len(distressed) * 2
stable = (remaining[remaining['label'] == 0]
          .sample(n=num_stable, random_state=SEED).copy())

balanced_parcels = (pd.concat([distressed, stable])
                    .sample(frac=1, random_state=SEED)
                    .reset_index(drop=True))
balanced_parcels['label'] = balanced_parcels['label'].replace({2: 1})

print(f"Balanced training pool: {len(balanced_parcels):,}")
print(balanced_parcels['label'].value_counts())

(DRIVE_BASE / 'data' / 'balanced_parcels.geojson').unlink(missing_ok=True)
balanced_parcels[['geometry', 'label']].to_file(
    DRIVE_BASE / 'data' / 'balanced_parcels.geojson', driver='GeoJSON'
)"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("## 4. Clay checkpoint + metadata"))

cells.append(code("""import urllib.request

CKPT_PATH = DRIVE_BASE / "data" / "clay-v1.5.ckpt"
METADATA_PATH = DRIVE_BASE / "data" / "clay_metadata.yaml"

if not CKPT_PATH.exists():
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Clay v1.5 checkpoint (~3 GB)…")
    urllib.request.urlretrieve(
        "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt",
        CKPT_PATH,
    )
else:
    print(f"Clay checkpoint: {CKPT_PATH} ({CKPT_PATH.stat().st_size/1e9:.1f} GB)")

if not METADATA_PATH.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml",
        METADATA_PATH,
    )
print(f"Clay metadata: {METADATA_PATH}")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 5. Chip cache — 130 ft (~40 m) fixed windows

ConvNeXt's 224×224 pad+resize logic maps to Clay's 256×256 in the dataset.
Existing 130-ft chips from previous sessions work — no rechip needed."""))

cells.append(code("""from tqdm import tqdm

CHIP_FEET = 130.0


def read_fixed_window(src, geom, size_feet: float = CHIP_FEET):
    cx, cy = geom.centroid.x, geom.centroid.y
    half = size_feet / 2
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

cells.append(code("""REBUILD_CHIPS = False

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
    print(f"Precaching {needed:,} chips from S3 (~20-30 min)…")
    precache_chips(balanced_parcels, src, CHIP_DIR)

if not DRIVE_TAR.exists() and IS_COLAB:
    print(f"Creating tarball for future sessions…")
    os.system(f'tar -cf "{LOCAL_TAR}" -C /content chips_local')
    os.system(f'cp "{LOCAL_TAR}" "{DRIVE_TAR}"')"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 6. Dataset — 256×256 output for Clay's native resolution"""))

cells.append(code("""IMAGE_SIZE = 256  # Clay's native pretraining resolution

# LINZ band stats from Clay's metadata.yaml — 0.5m aerial RGB
CLAY_RGB_MEAN = torch.tensor([89.96, 99.46, 89.51]).view(3, 1, 1) / 255.0
CLAY_RGB_STD  = torch.tensor([41.83, 36.96, 31.45]).view(3, 1, 1) / 255.0


def _pad_to_square_and_resize(arr, size=IMAGE_SIZE):
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
    \"\"\"Minimal dataset for embedding extraction — no augmentations needed.\"\"\"

    def __init__(self, gdf, chip_dir=None, label_col='label'):
        self.gdf = gdf
        self.chip_dir = Path(chip_dir) if chip_dir is not None else CHIP_DIR
        self.label_col = label_col

    def __len__(self):
        return len(self.gdf)

    def __getitem__(self, idx):
        row = self.gdf.iloc[idx]
        label = int(row[self.label_col])
        chip_id = row.name

        chip_path = self.chip_dir / f"{chip_id}.npy"
        try:
            arr = np.load(chip_path).astype(np.float32)
            arr = np.clip(arr / 255.0, 0.0, 1.0)
        except Exception:
            arr = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        img = _pad_to_square_and_resize(arr, size=IMAGE_SIZE)
        img = (img - CLAY_RGB_MEAN) / CLAY_RGB_STD
        return img, torch.tensor(label, dtype=torch.long)"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 7. Load Clay — official ClayMAEModule with prescribed args

Arguments match the [embeddings tutorial](https://clay-foundation.github.io/model/tutorials/embeddings.html) exactly."""))

cells.append(code("""from claymodel.module import ClayMAEModule

print("Loading Clay v1.5 large …")
clay_module = ClayMAEModule.load_from_checkpoint(
    checkpoint_path=str(CKPT_PATH),
    model_size="large",                                  # ← official arg from tutorial
    metadata_path=str(METADATA_PATH),
    dolls=[16, 32, 64, 128, 256, 768, 1024],             # ← MRL levels from tutorial
    doll_weights=[1, 1, 1, 1, 1, 1, 1],
    mask_ratio=0.0,
    shuffle=False,
    strict=False,
)
clay_module.eval()
clay_module = clay_module.to(DEVICE)

clay_encoder = clay_module.model.encoder
print(f"  encoder: {type(clay_encoder).__name__}")
print(f"  params: {sum(p.numel() for p in clay_encoder.parameters()):,}")

# LINZ-aligned conditioning (zero time/latlon per Clay docs)
RGB_WAVES = torch.tensor([0.635, 0.555, 0.465])
RGB_GSD   = torch.tensor(0.5)"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 8. Extract embeddings

Run Clay once per chip and cache the 1024-dim CLS token as a numpy array.
This is the "heavy" step — ~5–10 min on A100 for ~21K chips."""))

cells.append(code("""@torch.no_grad()
def extract_embeddings(loader, encoder, device=DEVICE):
    all_embeds = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Extracting"):
        images = images.to(device, non_blocking=True)
        B = images.shape[0]
        datacube = {
            "pixels": images,
            "time":   torch.zeros(B, 4, device=device),
            "latlon": torch.zeros(B, 4, device=device),
            "gsd":    RGB_GSD.to(device),
            "waves":  RGB_WAVES.to(device),
        }
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            embeddings, *_ = encoder(datacube)
            cls = embeddings[:, 0, :].float()          # CLS token, [B, 1024]
        all_embeds.append(cls.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_embeds), np.concatenate(all_labels)"""))

cells.append(code("""from sklearn.model_selection import train_test_split

# Use the full balanced pool — no augmentation, just pure feature extraction
full_loader = DataLoader(
    ParcelDataset(balanced_parcels),
    batch_size=64, shuffle=False, num_workers=0,
    pin_memory=(DEVICE == 'cuda'),
)

EMBED_PATH = DRIVE_BASE / "data" / "clay_embeddings_balanced.npz"

if EMBED_PATH.exists():
    print(f"Loading cached embeddings: {EMBED_PATH}")
    data = np.load(EMBED_PATH)
    X_all, y_all = data['X'], data['y']
else:
    print(f"Extracting Clay embeddings for {len(balanced_parcels):,} chips …")
    X_all, y_all = extract_embeddings(full_loader, clay_encoder)
    np.savez(EMBED_PATH, X=X_all, y=y_all)
    print(f"Saved: {EMBED_PATH}")

print(f"\\nEmbedding shape: {X_all.shape}")
print(f"Label shape:     {y_all.shape}")
print(f"Label balance:   {np.bincount(y_all)}")

# Feature sanity check — are embeddings discriminative?
std_across = X_all.std(axis=0).mean()
std_within = X_all.std(axis=1).mean()
print(f"\\nEmbedding quality:")
print(f"  std_within_sample = {std_within:.3f}")
print(f"  std_across_batch  = {std_across:.4f}  (need > 0.05 for classifier to work)")"""))

cells.append(code("""# 80/20 split on the balanced embeddings
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED,
)
print(f"Train: {X_train.shape}  Val: {X_val.shape}")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 9. Train RandomForest classifier

Matches the official [finetune-on-embeddings tutorial](https://clay-foundation.github.io/model/finetune/finetune-on-embeddings.html) exactly."""))

cells.append(code("""from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve,
)

# Tutorial uses defaults; we bump n_estimators for a slightly stronger model
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1,
)

print("Training RandomForest on Clay embeddings …")
rf.fit(X_train, y_train)
print("Done.")

# Default-threshold predictions (RF uses 0.5)
y_pred  = rf.predict(X_val)
y_proba = rf.predict_proba(X_val)[:, 1]   # probability of class 1

print(f"\\n--- Validation report (RF default threshold 0.5) ---")
print(classification_report(y_val, y_pred,
                            target_names=["Stable", "Distressed"], zero_division=0))

ConfusionMatrixDisplay.from_predictions(
    y_val, y_pred, display_labels=["Stable", "Distressed"], cmap="Blues",
)
plt.title("Validation Confusion Matrix")
plt.tight_layout()
plt.show()"""))

cells.append(md("""## 10. Threshold tuning"""))

cells.append(code("""precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = f1s[:-1].argmax()
THRESHOLD_TUNED = float(thresholds[best_idx])

print(f"Best-F1 threshold: {THRESHOLD_TUNED:.3f}")
print(f"  P={precisions[best_idx]:.3f}  R={recalls[best_idx]:.3f}  F1={f1s[best_idx]:.3f}\\n")

print(f"{'thr':>5}  {'recall':>7}  {'prec':>7}  {'f1':>6}  {'pos_rate':>8}")
for thr in [0.20, 0.30, 0.33, 0.40, 0.50, 0.60, 0.70]:
    p = (y_proba > thr).astype(int)
    tp = int(((p == 1) & (y_val == 1)).sum())
    fp = int(((p == 1) & (y_val == 0)).sum())
    fn = int(((p == 0) & (y_val == 1)).sum())
    rec  = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"{thr:5.2f}  {rec:7.3f}  {prec:7.3f}  {f1:6.3f}  {p.mean():8.2f}")

print(f"\\n--- At tuned threshold {THRESHOLD_TUNED:.3f} ---")
print(classification_report(y_val, (y_proba > THRESHOLD_TUNED).astype(int),
                            target_names=["Stable", "Distressed"], zero_division=0))"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 11. Natural-distribution holdout evaluation"""))

cells.append(code("""natural_chip_dir = Path("/content/natural_chips") if IS_COLAB else PROJECT_ROOT / "data" / "natural_chips"
natural_chip_dir.mkdir(parents=True, exist_ok=True)

print(f"Chipping {len(natural_test):,} natural-distribution parcels…")
precache_chips(natural_test, src, natural_chip_dir, desc="Natural chips")

NAT_EMBED_PATH = DRIVE_BASE / "data" / "clay_embeddings_natural.npz"

if NAT_EMBED_PATH.exists():
    data = np.load(NAT_EMBED_PATH)
    X_nat, y_nat = data['X'], data['y']
    print(f"Loaded cached: {NAT_EMBED_PATH}")
else:
    nat_loader = DataLoader(
        ParcelDataset(natural_test, chip_dir=natural_chip_dir, label_col='binary_label'),
        batch_size=64, shuffle=False, num_workers=0,
        pin_memory=(DEVICE == 'cuda'),
    )
    print(f"Extracting Clay embeddings for {len(natural_test):,} natural-dist chips …")
    X_nat, y_nat = extract_embeddings(nat_loader, clay_encoder)
    np.savez(NAT_EMBED_PATH, X=X_nat, y=y_nat)

y_nat_proba = rf.predict_proba(X_nat)[:, 1]
y_nat_pred  = (y_nat_proba > THRESHOLD_TUNED).astype(int)

print(f"\\nNatural-distribution test (threshold={THRESHOLD_TUNED:.3f}):")
print(f"  n={len(y_nat):,}  true prior={np.mean(y_nat):.4f}")
print(f"  predicted positive rate: {np.mean(y_nat_pred):.4f}")
print(classification_report(y_nat, y_nat_pred,
                            target_names=["Stable", "Distressed"], zero_division=0))

ConfusionMatrixDisplay.from_predictions(
    y_nat, y_nat_pred,
    display_labels=["Stable", "Distressed"], cmap="Oranges",
)
plt.title("Natural-Distribution Confusion Matrix")
plt.tight_layout()
plt.show()"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 12. Feature importance — which dimensions matter"""))

cells.append(code("""importances = rf.feature_importances_
top = np.argsort(importances)[::-1][:20]

print("Top 20 most important Clay embedding dimensions:")
print(f"{'rank':>4}  {'dim':>4}  {'importance':>10}")
for i, d in enumerate(top):
    print(f"{i+1:4d}  {d:4d}  {importances[d]:10.4f}")

fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(range(len(importances)), sorted(importances, reverse=True))
ax.set_xlabel("Rank")
ax.set_ylabel("Importance")
ax.set_title("Clay embedding dimension importances (sorted)")
plt.tight_layout()
plt.show()"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 13. Save the fitted RandomForest"""))

cells.append(code("""import joblib

RF_PATH = DRIVE_BASE / "data" / "clay_rf_classifier.joblib"
joblib.dump(rf, RF_PATH)
print(f"Saved RandomForest → {RF_PATH}")
print(f"  size: {RF_PATH.stat().st_size / (1024*1024):.1f} MB")
print(f"  threshold: {THRESHOLD_TUNED:.3f}")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 14. Full-city inference (OPTIONAL)

Gated behind `RUN_FULL_CITY = False`. Precaches ~547K chips (~30 min),
extracts Clay embeddings for all of them (~30 min on A100), then runs
RandomForest inference (~1 min)."""))

cells.append(code("""RUN_FULL_CITY = False

if not RUN_FULL_CITY:
    print("Skipping full-city inference. Set RUN_FULL_CITY=True to run.")
else:
    full_chip_dir = Path("/content/full_chips") if IS_COLAB else PROJECT_ROOT / "data" / "full_chips"
    full_chip_dir.mkdir(parents=True, exist_ok=True)

    full_df = parcels_labeled.copy().reset_index(drop=True)
    full_df['binary_label'] = (full_df['label'] > 0).astype(int)

    print(f"Chipping {len(full_df):,} parcels (~30 min)…")
    precache_chips(full_df, src, full_chip_dir, desc="Full-city chips")

    FULL_EMBED_PATH = DRIVE_BASE / "data" / "clay_embeddings_fullcity.npz"

    if FULL_EMBED_PATH.exists():
        data = np.load(FULL_EMBED_PATH)
        X_full = data['X']
    else:
        full_loader = DataLoader(
            ParcelDataset(full_df, chip_dir=full_chip_dir, label_col='binary_label'),
            batch_size=64, shuffle=False, num_workers=0,
            pin_memory=(DEVICE == 'cuda'),
        )
        print(f"Extracting Clay embeddings for {len(full_df):,} parcels …")
        X_full, _ = extract_embeddings(full_loader, clay_encoder)
        np.savez(FULL_EMBED_PATH, X=X_full)

    distress_proba = rf.predict_proba(X_full)[:, 1]
    full_df["distress_score"]  = distress_proba
    full_df["pred_distressed"] = (distress_proba > THRESHOLD_TUNED).astype(int)

    # Permit filter
    full_df.loc[full_df["label_permit_flagged"], "pred_distressed"] = 0

    print("\\nPrediction counts after permit filter:")
    print(full_df["pred_distressed"].value_counts())

    out_path = DRIVE_BASE / "output" / "predictions.geojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["geometry", "parcel_id", "distress_score", "pred_distressed", "label"]
    full_df[cols].to_file(out_path, driver="GeoJSON")
    print(f"\\nSaved: {out_path}")"""))

# ─────────────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

out_path = Path("F:/GitHub/musa6500-finalproject/main_v4.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path}")
print(f"Cells: {len(cells)} "
      f"({sum(1 for c in cells if c.cell_type == 'code')} code, "
      f"{sum(1 for c in cells if c.cell_type == 'markdown')} markdown)")
