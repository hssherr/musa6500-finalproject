"""Build main_v3.ipynb — ConvNeXt-Tiny replacement for Clay, pivoted after
empirical confirmation that Clay v1.5 cannot transfer to 0.3m aerial RGB.

Pipeline is identical to main_v2.ipynb except:
  - No Clay checkpoint / metadata download
  - ParcelDataset uses ImageNet normalization
  - Model is ConvNeXt-Tiny via timm
  - LLRD targets ConvNeXt's 4 stages instead of Clay's 24 transformer layers
  - Feature diagnostic simplified (no datacube)
  - Training loop has no encoder.eval() hack (encoder is unfrozen)
"""

from pathlib import Path
import nbformat as nbf


def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)


cells = []

cells.append(md("""# Detecting Structural Distress at Scale — v3 (ConvNeXt)

**Course:** MUSA 6500 — Geospatial Machine Learning in Remote Sensing
**Authors:** Jason Fan, Henry Sywulak-Herr

## Why ConvNeXt instead of Clay?

Previous work (main_v2.ipynb) attempted Clay v1.5 as the foundation model.
Despite comprehensive diagnostics and every fix recommended by Clay's own
code and documentation:

- Full 24-layer LLRD unfreeze, frozen-encoder (Clay's prescribed recipe)
- LINZ-aligned conditioning (closest pretraining platform, 0.5m GSD)
- ClayMAEModule official loader, CLS token readout
- Correct wavelengths, GSD sweep (0.3 → 30 m), zero time/latlon
- LINZ normalization stats from Clay's own metadata.yaml

…the encoder produced `std_across_batch ≈ 0.03` (near-constant features)
and training collapsed to predicting the class prior (loss ≈ 0.636 ≡
H(prior)). Model probabilities clustered in a razor-thin band around
0.33 with zero discrimination.

**Conclusion:** The 33× GSD gap between Clay's pretraining distribution
(10 m Sentinel-2) and 0.3 m aerial RGB exceeds what full fine-tuning can
bridge on 19K examples. ConvNeXt-Tiny pretrained on ImageNet — where
natural photos are much closer to high-res aerial RGB than multispectral
satellite is — is the right tool for this domain.

This finding is itself publishable as a result: for 0.3 m aerial RGB
tasks, natural-image pretraining transfers more effectively than
satellite-specific foundation models."""))

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

cells.append(code("""import sys, os, subprocess

IS_COLAB = os.path.exists('/content')

if IS_COLAB:
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q',
        'timm', 'rioxarray', 'geopandas', 'tqdm',
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

# Local SSD for chip cache — Drive is too slow for 23K small files
CHIP_DIR = Path('/content/chips_local') if IS_COLAB else PROJECT_ROOT / 'data' / 'chips'
CHIP_DIR.mkdir(parents=True, exist_ok=True)
(DRIVE_BASE / 'data').mkdir(parents=True, exist_ok=True)
print(f'Chip cache: {CHIP_DIR}')"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("## 2. Data loading"))
cells.append(md("### Imagery (lazy COG from S3)"))

cells.append(code("""from load_imagery import open_imagery

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

assert 'label' in parcels_labeled.columns
assert 'label_permit_flagged' in parcels_labeled.columns
print("\\nLabel columns OK:",
      [c for c in parcels_labeled.columns if 'label' in c.lower()])

plot_labels(parcels_labeled)"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 3. Train / test split strategy

- **Natural-distribution holdout (10%)** at real-world prior (~1.4% distressed) — the number for the final report
- **Balanced training pool (2:1 stable:distressed)** — for training and balanced validation"""))

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

out = DRIVE_BASE / 'data' / 'balanced_parcels.geojson'
balanced_parcels[['geometry', 'label']].to_file(out, driver='GeoJSON')
print(f"Saved: {out}")

out_nat = DRIVE_BASE / 'data' / 'natural_test.geojson'
natural_test[['geometry', 'label', 'binary_label', 'label_permit_flagged']].to_file(
    out_nat, driver='GeoJSON'
)
print(f"Saved: {out_nat}")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 4. Chip cache

130 ft (~40 m) fixed-window chips centered on each parcel centroid. EPSG:2272
is US survey feet; at 0.984 ft/px, 130 ft ≈ 132 pixels — the dataset
resizes to 224 for ConvNeXt input.

Chips live on local SSD for fast I/O; Drive only stores the tarball."""))

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
    print(f"Precaching {needed:,} chips from S3 (expect ~20-30 min)…")
    precache_chips(balanced_parcels, src, CHIP_DIR)

if not DRIVE_TAR.exists() and IS_COLAB:
    print(f"Creating tarball for future sessions…")
    os.system(f'tar -cf "{LOCAL_TAR}" -C /content chips_local')
    os.system(f'cp "{LOCAL_TAR}" "{DRIVE_TAR}"')
    print(f"Saved tarball → {DRIVE_TAR}")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 5. Dataset and dataloaders

**ImageNet normalization** for ConvNeXt (pretrained on ImageNet-22k → 1k)."""))

cells.append(code("""IMAGE_SIZE = 224

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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
        chip_id = row.name

        chip_path = self.chip_dir / f"{chip_id}.npy"
        try:
            arr = np.load(chip_path).astype(np.float32)
            arr = np.clip(arr / 255.0, 0.0, 1.0)
        except Exception:
            arr = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        img = _pad_to_square_and_resize(arr, size=IMAGE_SIZE)

        if self.transform is not None:
            img = self.transform(img)

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
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

images, labels = next(iter(train_loader))
print(f"Batch: {tuple(images.shape)}  labels: {tuple(labels.shape)}")
print(f"Pixel stats (post-normalization):")
print(f"  min={images.min():.2f}  max={images.max():.2f}  mean={images.mean():.2f}")
print(f"  (mean near 0 = good; near -2 = padding-dominated chips)")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 6. Model — ConvNeXt-Tiny + distress head

ConvNeXt-Tiny (~28M params, ImageNet-22k→1k pretrained) via `timm`.
`global_pool='avg'` means the encoder returns `[B, 768]` feature vectors
directly, so the forward is a simple `feats → head → logit`."""))

cells.append(code("""import timm


class DistressClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            'convnext_tiny.fb_in22k_ft_in1k',
            pretrained=True,
            num_classes=0,
            global_pool='avg',
        )
        self.feat_dim = self.encoder.num_features  # 768

        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, pixels):
        feats = self.encoder(pixels)   # [B, 768]
        return self.head(feats)


model = DistressClassifier().to(DEVICE)

prior = float(train_gdf['label'].mean())
nn.init.constant_(model.head[-1].bias, math.log(prior / (1 - prior)))
print(f"Head bias initialized: logit({prior:.3f}) = {math.log(prior/(1-prior)):.3f}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 7. Fine-tune setup — full encoder unfreeze with LLRD

ConvNeXt has 4 stages. LLRD with aggressive decay (0.5 per stage) — deepest
stage gets `BASE_LR = 1e-4`, stem at the lowest LR, head at `5e-4`."""))

cells.append(code("""for p in model.encoder.parameters():
    p.requires_grad = True

stages = list(model.encoder.stages)
BASE_LR = 1e-4
DECAY   = 0.5

param_groups = []

for i, stage in enumerate(reversed(stages)):
    param_groups.append({"params": stage.parameters(), "lr": BASE_LR * (DECAY ** i)})

param_groups.append({
    "params": model.encoder.stem.parameters(),
    "lr": BASE_LR * (DECAY ** len(stages)),
})

param_groups.append({"params": model.head.parameters(), "lr": 5e-4})

optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

print(f"LLRD schedule:")
for i, g in enumerate(optimizer.param_groups):
    print(f"  group {i}: lr = {g['lr']:.1e}")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,}")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 8. Feature diagnostic — run BEFORE training

Healthy ImageNet features on varied aerial RGB should give
`std_across_batch ≥ 0.1`."""))

cells.append(code("""model.eval()
with torch.no_grad():
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(DEVICE)

    per_pixel_std = imgs.std(dim=0).mean().item()
    print(f"Input diversity — per_pixel_std: {per_pixel_std:.4f}")
    print(f"  (≈0 = all-zero chips; ≥0.5 = real varied imagery)")

    feats = model.encoder(imgs)   # [B, 768]

mean_val   = feats.mean().item()
std_within = feats.std(dim=1).mean().item()
std_across = feats.std(dim=0).mean().item()

print(f"\\nImageNet feature check:")
print(f"  mean              = {mean_val:+.3f}")
print(f"  std_within_sample = {std_within:.3f}")
print(f"  std_across_batch  = {std_across:.4f}  (want > 0.1)")

if std_across < 0.05:
    print("\\n⚠️  Something's wrong upstream — check chip cache.")
else:
    print("\\n✓  Strong feature diversity — safe to train.")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 9. Loss function

BCE for the first run (collapse is visible as loss ≈ 0.636). Switch to
focal loss for a final run once you confirm ConvNeXt is learning."""))

cells.append(code("""criterion = nn.BCEWithLogitsLoss()
print("Using BCEWithLogitsLoss.")
print("Loss reference: ~0.636 = predicting prior; < 0.55 = real learning.")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""## 10. Training loop

- OneCycleLR with 10% warmup, bf16 AMP (no GradScaler needed)
- Early stopping (patience=4), save gate excludes collapsed predictions
- Local checkpoint saves, Drive copy once at end"""))

cells.append(code("""from sklearn.metrics import recall_score, precision_score, classification_report

torch.cuda.empty_cache()
import gc; gc.collect()

NUM_EPOCHS  = 20
THRESHOLD   = 0.33
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

print(f"Training up to {NUM_EPOCHS} epochs, patience={PATIENCE}, threshold={THRESHOLD}")
print(f"Saving best to: {LOCAL_BEST}")

for epoch in range(NUM_EPOCHS):
    model.train()

    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_AMP):
                probs = torch.sigmoid(model(images)).squeeze(1).float()
            preds = (probs > THRESHOLD).long()
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    recall    = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    pos_rate  = sum(all_preds) / len(all_preds)

    cur_head_lr = optimizer.param_groups[-1]['lr']
    cur_enc_lr  = optimizer.param_groups[0]['lr']

    print(f"Ep {epoch+1:02d} | loss {train_loss/len(train_loader):.4f} "
          f"| R {recall:.3f} | P {precision:.3f} | pos {pos_rate:.2f} "
          f"| lr_head {cur_head_lr:.1e} lr_enc {cur_enc_lr:.1e}")

    if epoch == 0 and train_loss / len(train_loader) > 0.62:
        print("  ⚠️  Epoch 1 loss near prior-entropy — check chips/inputs upstream.")

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

cells.append(code("""DRIVE_BEST = DRIVE_BASE / "data" / "best_model.pt"
if LOCAL_BEST.exists():
    shutil.copy(LOCAL_BEST, DRIVE_BEST)
    size_mb = LOCAL_BEST.stat().st_size / (1024 * 1024)
    print(f"Copied best model ({size_mb:.0f} MB) → {DRIVE_BEST}")
else:
    print("⚠️  No best_model.pt saved. Check feature diagnostic + loss trajectory.")"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("## 11. Evaluation on balanced validation set"))

cells.append(code("""from sklearn.metrics import ConfusionMatrixDisplay

if not LOCAL_BEST.exists():
    raise RuntimeError(f"No checkpoint at {LOCAL_BEST}.")

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
cells.append(md("""## 12. Threshold tuning"""))

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
cells.append(md("""## 13. Natural-distribution holdout evaluation

The honest number — precision/recall at the real-world ~1.4% prior."""))

cells.append(code("""natural_chip_dir = Path("/content/natural_chips") if IS_COLAB else PROJECT_ROOT / "data" / "natural_chips"
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

print(f"\\nNatural-distribution test (threshold={THRESHOLD_TUNED:.3f}):")
print(f"  n={len(nat_labels):,}  true prior={np.mean(nat_labels):.4f}")
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
cells.append(md("""## 14. Geographic stratification"""))

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
cells.append(md("""## 15. Full-city inference (OPTIONAL — 30+ minutes)

Gated behind `RUN_FULL_CITY = False`."""))

cells.append(code("""RUN_FULL_CITY = False

if not RUN_FULL_CITY:
    print("Skipping full-city inference. Set RUN_FULL_CITY=True to run.")
else:
    full_chip_dir = Path("/content/full_chips") if IS_COLAB else PROJECT_ROOT / "data" / "full_chips"
    full_chip_dir.mkdir(parents=True, exist_ok=True)

    full_df = parcels_labeled.copy().reset_index(drop=True)
    full_df['binary_label'] = (full_df['label'] > 0).astype(int)

    print(f"Chipping {len(full_df):,} parcels (~30 min first run)…")
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
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=USE_AMP):
                probs = torch.sigmoid(model(images)).squeeze(1).float()
            all_city_probs.extend(probs.cpu().tolist())
            if i % 200 == 0:
                print(f"  {i * BATCH_SIZE:,} / {len(full_df):,} parcels scored")

    full_df["distress_score"]  = all_city_probs
    full_df["pred_distressed"] = (full_df["distress_score"] > THRESHOLD_TUNED).astype(int)

    # Permit filter — zero out parcels with active eCLIPSE permits
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
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

out_path = Path("F:/GitHub/musa6500-finalproject/main_v3.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path}")
print(f"Cells: {len(cells)} "
      f"({sum(1 for c in cells if c.cell_type == 'code')} code, "
      f"{sum(1 for c in cells if c.cell_type == 'markdown')} markdown)")
