#!/usr/bin/env python
# coding: utf-8

# # Construction-Site PPE Detection & Compliance Monitoring
# 
# **Goal:** train a custom **YOLO11** detector to spot PPE violations on construction sites
# (missing helmets / vests), then wire it up to **OpenCV** for image and video inference with
# person-level compliance logic and temporal violation confirmation.
# 
# **Workflow:** dataset engineering → automated validation → curated + frozen dataset → YOLO11
# baseline training → quantitative evaluation → error analysis → OpenCV integration → video
# pipeline with PPE-compliance reasoning.
# 
# Dataset: Roboflow `construction-rineu` (YOLO format), 5 classes:
# `0 helmet · 1 no-helmet · 2 no-vest · 3 person · 4 vest`
# 

# ## Phase 0 — Download the dataset
# 
# Pulls the labeled dataset from Roboflow using a Kaggle secret (`RF_API`) so the API key never
# appears in the notebook.
# 

# In[1]:


get_ipython().system('pip install -q roboflow')

from kaggle_secrets import UserSecretsClient
from roboflow import Roboflow

api_key = UserSecretsClient().get_secret("RF_API")

rf = Roboflow(api_key=api_key)
project = rf.workspace("envisage").project("construction-rineu")
dataset = project.version(3).download("yolov11")

print("Dataset downloaded to:", dataset.location)


# ## Phase 1 — Quick dataset exploration
# 
# Sanity-check the folder layout, `data.yaml`, image counts/dimensions, and eyeball a handful of
# training images before doing anything else.
# 

# In[2]:


from pathlib import Path
from PIL import Image
from collections import Counter

DATASET_DIR = Path("/kaggle/working/construction-3")
SPLITS = ["train", "valid", "test"]

print(f"data.yaml:\n{(DATASET_DIR / 'data.yaml').read_text()}")

for split in SPLITS:
    images = list((DATASET_DIR / split / "images").glob("*"))
    labels = list((DATASET_DIR / split / "labels").glob("*.txt"))

    sizes = Counter(Image.open(p).size for p in images[:20])

    print(f"\n{split.upper()}: {len(images)} images, {len(labels)} label files")
    print("  sample dimensions:", dict(sizes))


# In[3]:


import random
import matplotlib.pyplot as plt

image_files = list((DATASET_DIR / "train" / "images").glob("*"))
samples = random.sample(image_files, 12)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for ax, image_path in zip(axes.ravel(), samples):
    ax.imshow(Image.open(image_path))
    ax.set_title(image_path.name, fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()


# ## Phase 2 — Automated YOLO label validation
# 
# Checks every label file for structural problems: invalid class IDs, malformed lines,
# out-of-range coordinates, and zero/negative box dimensions. Images with **no** label file, and
# label files with **zero** annotations, are legitimate negative/background examples and are
# **not** treated as errors — only genuinely malformed annotations are flagged.
# 

# In[4]:


import yaml
from pathlib import Path
from collections import Counter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

yaml_path = next(DATASET_DIR.glob("*.yaml"))
class_names = yaml.safe_load(yaml_path.read_text())["names"]
class_names = (
    {int(k): str(v) for k, v in class_names.items()}
    if isinstance(class_names, dict)
    else {i: str(n) for i, n in enumerate(class_names)}
)
NUM_CLASSES = len(class_names)
print("Classes:", class_names)


def validate_label_line(line, errors):
    """Validate one `class_id x_center y_center width height` line. Returns True if valid."""
    parts = line.split()
    if len(parts) != 5:
        errors.append(("Malformed line", f"expected 5 values, found {len(parts)}: {line}"))
        return False

    try:
        class_id = int(float(parts[0]))
        assert float(parts[0]).is_integer()
    except (ValueError, AssertionError):
        errors.append(("Invalid class ID", f"non-integer class id: {parts[0]}"))
        return False

    if not (0 <= class_id < NUM_CLASSES):
        errors.append(("Invalid class ID", f"class_id={class_id} outside 0-{NUM_CLASSES - 1}"))
        return False

    try:
        xc, yc, w, h = map(float, parts[1:])
    except ValueError:
        errors.append(("Malformed line", f"non-numeric coordinates: {line}"))
        return False

    if any(not (0 <= v <= 1) for v in (xc, yc, w, h)):
        errors.append(("Coordinate outside [0,1]", line))
        return False

    if w <= 0 or h <= 0:
        errors.append(("Zero/negative box size", f"w={w}, h={h}"))
        return False

    return True


def validate_split(split_dir, split_name):
    """Validate one dataset split and return an (image, label, error) stats dict."""
    image_dir, label_dir = split_dir / "images", split_dir / "labels"
    images = {p.stem: p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}
    labels = {p.stem: p for p in label_dir.glob("*.txt")}

    stats = Counter()
    errors = []
    stats["images"], stats["label_files"] = len(images), len(labels)
    stats["images_without_labels"] = len(images.keys() - labels.keys())  # legitimate negatives
    stats["orphan_labels"] = len(labels.keys() - images.keys())

    for stem, label_path in labels.items():
        if stem not in images:
            continue  # already counted as orphan
        lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
        if not lines:
            stats["empty_label_files"] += 1  # legitimate negative
            continue
        line_errors = []
        for line in lines:
            validate_label_line(line, line_errors)
        if line_errors:
            errors.extend((split_name, label_path.name, t, d) for t, d in line_errors)
        else:
            stats["valid_label_files"] += 1

    return stats, errors


all_errors = []
print("\n" + "=" * 70)
for split in SPLITS:
    stats, errors = validate_split(DATASET_DIR / split, split)
    all_errors.extend(errors)
    print(f"{split.upper():<6} -> {dict(stats)}")

print("\n" + "=" * 70)
if not all_errors:
    print("RESULT: PASS  — no malformed annotations found.")
else:
    print(f"RESULT: FAIL  — {len(all_errors)} issue(s) found:")
    for split, file, err_type, detail in all_errors[:50]:
        print(f"  [{split}] {file}: {err_type} — {detail}")


# ## Phase 3 — Apply confirmed dataset-curation fixes
# 
# Manual inspection (already performed) confirmed these data-quality issues, each backed by a
# specific reason rather than a blind hash-based delete:
# 
# | Issue | Resolution |
# |---|---|
# | `images-6-*.jpg` exists in both **train** and **valid** (exact duplicate) | keep the **valid** copy, remove it from **train** |
# | Two copies of `back141*` / `download-4*` | keep `back141*`, remove `download-4*` (empty label) |
# | Two copies of a `19461*` image | keep the first copy, remove the second (malformed/commented annotation) |
# | Two copies of an `istockphoto-1125653149*` image | keep the first copy, remove the second (missing a `no-vest` annotation) |
# 
# `find_by_prefix` locates files by their documented prefix rather than a hard-coded hash suffix,
# so this cell stays correct even if Roboflow re-exports with different hash strings.
# 

# In[5]:


def find_by_prefix(directory, prefix, exclude=None):
    """Return files in `directory` whose name starts with `prefix` (case-insensitive)."""
    matches = sorted(p for p in directory.glob(f"{prefix}*") if p != exclude)
    return matches


def remove_pair(label_path):
    """Remove a label file together with its matching image, if present."""
    if label_path is None or not label_path.exists():
        return
    for ext in IMAGE_EXTENSIONS:
        image_path = label_path.with_suffix(ext)
        if image_path.exists():
            image_path.unlink()
            print(f"  removed image: {image_path.name}")
    label_path.unlink()
    print(f"  removed label: {label_path.name}")


train_labels = DATASET_DIR / "train" / "labels"
valid_labels = DATASET_DIR / "valid" / "labels"

print("1) Cross-split duplicate `images-6-*`: keep VALID copy, remove TRAIN copy")
for label_path in find_by_prefix(train_labels, "images-6-"):
    remove_pair(label_path)

print("\n2) `back141*` vs `download-4*`: keep back141, remove download-4 (empty label)")
for label_path in find_by_prefix(train_labels, "download-4"):
    remove_pair(label_path)

print("\n3) Duplicate `19461*`: keep first copy, remove second (malformed annotation)")
dupes_19461 = find_by_prefix(train_labels, "19461")
for label_path in dupes_19461[1:]:
    remove_pair(label_path)

print("\n4) Duplicate `istockphoto-1125653149*`: keep first, remove second (missing no-vest box)")
dupes_istock = find_by_prefix(train_labels, "istockphoto-1125653149")
for label_path in dupes_istock[1:]:
    remove_pair(label_path)

print("\nCuration fixes applied.")


# ## Phase 4 — Freeze the cleaned dataset as `construction_v1`
# 
# Everything from here on (training, evaluation, error analysis) reads from this frozen copy so
# the dataset never silently changes mid-experiment.
# 

# In[6]:


import shutil

FROZEN_DIR = Path("/kaggle/working/construction_v1")

if FROZEN_DIR.exists():
    shutil.rmtree(FROZEN_DIR)
shutil.copytree(DATASET_DIR, FROZEN_DIR)

DATA_YAML = FROZEN_DIR / "data.yaml"
assert DATA_YAML.exists(), f"data.yaml missing: {DATA_YAML}"

for split in SPLITS:
    n_images = len(list((FROZEN_DIR / split / "images").iterdir()))
    n_labels = len(list((FROZEN_DIR / split / "labels").iterdir()))
    print(f"{split.upper():<6} images: {n_images:<6} labels: {n_labels:<6}")

print("\nDataset frozen at:", FROZEN_DIR)


# ## Phase 5 — Environment setup
# 
# Confirm the GPU(s) are visible to PyTorch, then install and verify Ultralytics.
# 

# In[7]:


get_ipython().system('pip install -q -U ultralytics')

import torch
import ultralytics
from ultralytics import YOLO

print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("Ultralytics:", ultralytics.__version__)
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


# ## Phase 6 — YOLO11m baseline training
# 
# Deliberately a **simple baseline** — pretrained YOLO11m, no hyperparameter search yet — so it
# gives a fair reference point before any tuning. Checkpoints are saved every 5 epochs to enable
# picking the best periodic checkpoint later (Phase 8).
# 

# In[8]:


MODEL_NAME = "yolo11m.pt"
IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16
SEED = 42
DEVICE = 0

PROJECT_DIR = Path("/kaggle/working/yolo_runs")
RUN_NAME = "construction_v1_yolo11m_baseline"

print(f"Model={MODEL_NAME}  epochs={EPOCHS}  imgsz={IMG_SIZE}  batch={BATCH_SIZE}")

model = YOLO(MODEL_NAME)

results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    seed=SEED,
    deterministic=True,
    device=DEVICE,
    workers=4,
    pretrained=True,
    val=True,
    save=True,
    save_period=5,
    project=str(PROJECT_DIR),
    name=RUN_NAME,
    plots=True,
    verbose=True,
)

RUN_DIR = PROJECT_DIR / RUN_NAME
WEIGHTS_DIR = RUN_DIR / "weights"
BEST_PT, LAST_PT = WEIGHTS_DIR / "best.pt", WEIGHTS_DIR / "last.pt"

print("\nRun directory:", RUN_DIR)
print("Saved weights:", [w.name for w in sorted(WEIGHTS_DIR.glob("*.pt"))])


# ## Phase 7 — Evaluation: precision, recall, mAP, per-class AP, curves
# 
# One `model.val()` call on `best.pt` produces everything: overall metrics, per-class AP,
# confusion matrix and PR curves (all written under `RUN_DIR`) — no need to re-run validation
# per plot.
# 

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt

model = YOLO(str(BEST_PT))

val_metrics = model.val(
    data=str(DATA_YAML), split="val", imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE, plots=True
)

print(f"mAP50={val_metrics.box.map50:.4f}  mAP50-95={val_metrics.box.map:.4f} "
      f"Precision={val_metrics.box.mp:.4f}  Recall={val_metrics.box.mr:.4f}")
print("Confusion matrix / PR curves saved under:", RUN_DIR)

per_class_df = pd.DataFrame([
    {
        "Class": name,
        "Precision": val_metrics.box.p[i],
        "Recall": val_metrics.box.r[i],
        "mAP50": val_metrics.box.ap50[i],
        "mAP50-95": val_metrics.box.ap[i],
    }
    for i, name in class_names.items()
])
display(per_class_df.style.format({c: "{:.4f}" for c in per_class_df.columns[1:]}))


# In[10]:


df = pd.read_csv(RUN_DIR / "results.csv")
df.columns = df.columns.str.strip()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
axes[0].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
axes[0].set(xlabel="Epoch", ylabel="Metric", title="Validation performance")
axes[0].legend(); axes[0].grid(True)

for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
    if col in df.columns:
        axes[1].plot(df["epoch"], df[col], label=col)
axes[1].set(xlabel="Epoch", ylabel="Loss", title="Training loss")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.show()


# ## Phase 8 — Compare periodic checkpoints, pick the best, evaluate on test
# 
# `best.pt` (chosen by Ultralytics on fitness) isn't necessarily the periodic checkpoint with the
# highest `mAP50-95` on our validation split — so the epoch* checkpoints saved every 5 epochs are
# compared directly, and the winner becomes the model used for everything downstream.
# 

# In[11]:


import re

def epoch_number(path):
    match = re.search(r"epoch(\d+)", path.name)
    return int(match.group(1)) if match else -1

checkpoint_rows = []
for ckpt in sorted(WEIGHTS_DIR.glob("epoch*.pt"), key=epoch_number):
    metrics = YOLO(str(ckpt)).val(
        data=str(DATA_YAML), split="val", imgsz=IMG_SIZE, batch=BATCH_SIZE,
        device=DEVICE, plots=False, verbose=False,
    )
    checkpoint_rows.append({
        "Epoch": epoch_number(ckpt), "Precision": metrics.box.mp, "Recall": metrics.box.mr,
        "mAP50": metrics.box.map50, "mAP50-95": metrics.box.map,
    })

checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values("Epoch")
display(checkpoint_df.style.format({c: "{:.4f}" for c in checkpoint_df.columns[1:]}))

best_row = checkpoint_df.loc[checkpoint_df["mAP50-95"].idxmax()]
BEST_CHECKPOINT = WEIGHTS_DIR / f"epoch{int(best_row['Epoch'])}.pt"
print(f"\nBest periodic checkpoint: epoch {int(best_row.Epoch)}  "
      f"(mAP50-95={best_row['mAP50-95']:.4f}) -> {BEST_CHECKPOINT}")

plt.figure(figsize=(9, 5))
plt.plot(checkpoint_df["Epoch"], checkpoint_df["mAP50"], marker="o", label="mAP50")
plt.plot(checkpoint_df["Epoch"], checkpoint_df["mAP50-95"], marker="o", label="mAP50-95")
plt.xlabel("Epoch"); plt.ylabel("Validation metric"); plt.legend(); plt.grid(True)
plt.title("Checkpoint performance"); plt.show()


# In[12]:


FINAL_MODEL = YOLO(str(BEST_CHECKPOINT))

test_metrics = FINAL_MODEL.val(
    data=str(DATA_YAML), split="test", imgsz=IMG_SIZE, batch=BATCH_SIZE, device=DEVICE, plots=True
)

print("FINAL TEST RESULTS")
print(f"Precision={test_metrics.box.mp:.4f}  Recall={test_metrics.box.mr:.4f}  "
      f"mAP50={test_metrics.box.map50:.4f}  mAP50-95={test_metrics.box.map:.4f}")


# ## Phase 9 — Error analysis
# 
# Run the final model on the **test** split, match every prediction to ground truth by IoU, and
# bucket the misses into **false positives**, **false negatives**, and **class confusions** (the
# PPE-specific case — helmet ↔ no-helmet, vest ↔ no-vest — is pulled out separately since that's
# the failure mode that matters most for a compliance system).
# 

# In[13]:


import cv2
import numpy as np

TEST_IMAGES, TEST_LABELS = FROZEN_DIR / "test" / "images", FROZEN_DIR / "test" / "labels"
CLASS_NAMES = class_names  # {0: "helmet", 1: "no-helmet", 2: "no-vest", 3: "person", 4: "vest"}

results = FINAL_MODEL.predict(
    source=str(TEST_IMAGES), imgsz=IMG_SIZE, conf=0.25, iou=0.7, device=DEVICE,
    save=False, verbose=False,
)
print(f"Images processed: {len(results)}")


def load_ground_truth(label_path):
    if not label_path.exists():
        return []
    gt = []
    for line in label_path.read_text().splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        cls, xc, yc, w, h = map(float, values)
        gt.append({"class_id": int(cls), "class_name": CLASS_NAMES[int(cls)], "xc": xc, "yc": yc, "w": w, "h": h})
    return gt


records = []
for result in results:
    image_path = Path(result.path)
    gt = load_ground_truth(TEST_LABELS / f"{image_path.stem}.txt")

    preds = []
    if result.boxes is not None:
        for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
            preds.append({"class_id": int(cls), "class_name": CLASS_NAMES[int(cls)], "confidence": float(conf), "box": box.tolist()})

    records.append({"image": image_path, "ground_truth": gt, "predictions": preds})

print(f"Records created: {len(records)}")


# In[14]:


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def gt_to_xyxy(obj, width, height):
    xc, yc, w, h = obj["xc"] * width, obj["yc"] * height, obj["w"] * width, obj["h"] * height
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


IOU_THRESHOLD = 0.5
error_records = []

for record in records:
    image = cv2.imread(str(record["image"]))
    if image is None:
        continue
    height, width = image.shape[:2]

    gt_objects = [{**o, "box": gt_to_xyxy(o, width, height), "matched": False} for o in record["ground_truth"]]
    pred_objects = [{**p, "matched": False} for p in record["predictions"]]

    for pred in sorted(pred_objects, key=lambda p: p["confidence"], reverse=True):
        best_iou, best_gt = 0, None
        for gt in gt_objects:
            if gt["matched"]:
                continue
            iou = calculate_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou, best_gt = iou, gt

        if best_gt is not None and best_iou >= IOU_THRESHOLD:
            pred["matched"] = best_gt["matched"] = True
            if pred["class_id"] != best_gt["class_id"]:
                error_records.append({"image": record["image"], "type": "class_confusion",
                                       "ground_truth": best_gt["class_name"], "prediction": pred["class_name"],
                                       "confidence": pred["confidence"], "iou": best_iou})
        else:
            error_records.append({"image": record["image"], "type": "false_positive", "ground_truth": None,
                                   "prediction": pred["class_name"], "confidence": pred["confidence"], "iou": best_iou})

    for gt in gt_objects:
        if not gt["matched"]:
            error_records.append({"image": record["image"], "type": "false_negative", "ground_truth": gt["class_name"],
                                   "prediction": None, "confidence": None, "iou": 0})

error_df = pd.DataFrame(error_records)
print(f"Total error records: {len(error_df)}")
print("\nBy type:\n", error_df["type"].value_counts())


# In[15]:


# PPE-specific confusions: the failure mode that actually matters for compliance
confusion_df = error_df[error_df["type"] == "class_confusion"]
ppe_pairs = {("helmet", "no-helmet"), ("no-helmet", "helmet"), ("vest", "no-vest"), ("no-vest", "vest")}

ppe_confusions = confusion_df[
    confusion_df.apply(lambda r: (r["ground_truth"], r["prediction"]) in ppe_pairs, axis=1)
]
print(f"PPE confusion cases: {len(ppe_confusions)}")
display(ppe_confusions[["image", "ground_truth", "prediction", "confidence", "iou"]])


# In[16]:


def visualize_errors(error_df, error_type=None, max_images=8):
    """Show up to `max_images` example failures, optionally filtered by `error_type`."""
    subset = error_df if error_type is None else error_df[error_df["type"] == error_type]
    for _, row in subset.head(max_images).iterrows():
        image = cv2.cvtColor(cv2.imread(str(row["image"])), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f"{row['type']} | GT: {row['ground_truth']} | Pred: {row['prediction']}")
        plt.axis("off")
        plt.show()

# Example: inspect PPE class-confusion cases first, since they matter most
visualize_errors(error_df, error_type="class_confusion", max_images=6)


# In[17]:


ERROR_DIR = RUN_DIR / "error_analysis"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
error_df.to_csv(ERROR_DIR / "error_records.csv", index=False)
print("Saved:", ERROR_DIR / "error_records.csv")


# ## Phase 10 — OpenCV integration: PPE association & compliance logic
# 
# Reusable building blocks for the final application:
# - **`associate_ppe_to_person`** — links helmet/vest detections to a specific person using
#   head/torso regions (not just "nearest box"), since a raw detection list has no notion of
#   *whose* helmet is whose.
# - **`determine_compliance`** — turns that association into a `COMPLIANT` / `VIOLATION` /
#   `UNKNOWN` status per person.
# - **drawing helpers** — consistent OpenCV overlays used by every inference cell below.
# 

# In[18]:


CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

PERSON_CLASS, HELMET_CLASS, NO_HELMET_CLASS, NO_VEST_CLASS, VEST_CLASS = 3, 0, 1, 2, 4


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_inside_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def associate_ppe_to_person(person_box, ppe_detections):
    """Associate helmet/vest detections with one person via head/torso region containment."""
    px1, py1, px2, py2 = person_box
    person_height = py2 - py1

    head_region = (px1, py1, px2, py1 + person_height * 0.40)
    torso_region = (px1, py1 + person_height * 0.20, px2, py1 + person_height * 0.80)

    best = {"helmet": None, "no-helmet": None, "vest": None, "no-vest": None}
    region_for = {HELMET_CLASS: (head_region, "helmet"), NO_HELMET_CLASS: (head_region, "no-helmet"),
                  VEST_CLASS: (torso_region, "vest"), NO_VEST_CLASS: (torso_region, "no-vest")}

    for det in ppe_detections:
        region_info = region_for.get(det["class_id"])
        if region_info is None:
            continue
        region, key = region_info
        if point_inside_box(box_center(det["box"]), region):
            if best[key] is None or det["confidence"] > best[key]["confidence"]:
                best[key] = det
    return best


def determine_compliance(associated):
    helmet_status = "helmet" if associated["helmet"] else ("no-helmet" if associated["no-helmet"] else "unknown")
    vest_status = "vest" if associated["vest"] else ("no-vest" if associated["no-vest"] else "unknown")

    if helmet_status == "helmet" and vest_status == "vest":
        compliance = "COMPLIANT"
    elif helmet_status == "no-helmet" or vest_status == "no-vest":
        compliance = "VIOLATION"
    else:
        compliance = "UNKNOWN"
    return {"helmet": helmet_status, "vest": vest_status, "compliance": compliance}


def draw_box(frame, box, label, color=(255, 255, 255)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_person_status(frame, person_box, person_id, compliance):
    x1, y1, _, _ = map(int, person_box)
    color = {"COMPLIANT": (0, 200, 0), "VIOLATION": (0, 0, 255)}.get(compliance["compliance"], (255, 255, 255))
    label = f"Person {person_id}: {compliance['compliance']}"
    cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


# ### Quick sanity check on a single image
# 
# Runs the full association + compliance pipeline (not just YOLO's default box overlay) on one
# test image before moving to video.
# 

# In[19]:


sample_image_path = next(TEST_IMAGES.glob("*"))
frame = cv2.imread(str(sample_image_path))

result = FINAL_MODEL.predict(source=frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=DEVICE, verbose=False)[0]

detections = []
if result.boxes is not None:
    for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
        detections.append({"class_id": int(cls), "class_name": CLASS_NAMES[int(cls)], "confidence": float(conf), "box": box.tolist()})

persons = [d for d in detections if d["class_id"] == PERSON_CLASS]
ppe = [d for d in detections if d["class_id"] != PERSON_CLASS]

for det in ppe:
    draw_box(frame, det["box"], f"{det['class_name']} {det['confidence']:.2f}")
for i, person in enumerate(persons, start=1):
    compliance = determine_compliance(associate_ppe_to_person(person["box"], ppe))
    draw_box(frame, person["box"], f"person {person['confidence']:.2f}")
    draw_person_status(frame, person["box"], i, compliance)

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title(sample_image_path.name)
plt.show()


# ## Phase 11 — Full video pipeline: tracking + compliance + temporal violation events
# 
# This is the final application logic, combining everything above into one pass over a video:
# 
# 1. **Detect** every frame with YOLO11.
# 2. **Track** persons frame-to-frame with simple IoU matching (no ID switches from re-detecting
#    the same person as "new" every frame).
# 3. **Associate** PPE to each person and derive a compliance status.
# 4. **Confirm violations temporally** — a single missed-detection frame does not raise an alert;
#    a violation only becomes an **event** once it persists for `CONFIRMATION_FRAMES` in a row, and
#    the event stays open until `EVENT_END_FRAMES` clear frames pass. This is what prevents one
#    real violation from generating hundreds of duplicate alerts.
# 
# Set `INPUT_VIDEO` below to a real video path before running.
# 

# In[20]:


import traceback

def open_video_safe(path):
    """Try multiple backends before giving up. Returns an opened VideoCapture or None."""
    path = str(path)
    if not os.path.exists(path):
        print(f"[open_video_safe] File does not exist: {path}")
        return None
    if os.path.getsize(path) == 0:
        print(f"[open_video_safe] File exists but is 0 bytes: {path}")
        return None

    for backend, label in [(cv2.CAP_ANY, "default"), (cv2.CAP_FFMPEG, "CAP_FFMPEG")]:
        cap = cv2.VideoCapture(path, backend)
        if cap.isOpened():
            print(f"[open_video_safe] Opened successfully with backend: {label}")
            return cap
        cap.release()

    print(f"[open_video_safe] Could not open with any backend: {path}")
    return None


def run_ppe_pipeline(input_video, output_video, model, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=DEVICE):
    cap = open_video_safe(input_video)
    if cap is None:
        print("Aborting run_ppe_pipeline: no usable video source.")
        return []

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print("Could not create output video writer. Aborting.")
        cap.release()
        return []

    tracker, event_log = PersonTracker(), ViolationEventLog()
    frame_number, start_time = 0, time.time()
    skipped_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1

            try:
                result = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
                detections = []
                if result.boxes is not None:
                    for box, cls, confidence in zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.cls.cpu().numpy().astype(int),
                        result.boxes.conf.cpu().numpy(),
                    ):
                        detections.append({"class_id": int(cls), "class_name": CLASS_NAMES[int(cls)],
                                            "confidence": float(confidence), "box": box.tolist()})

                persons_raw = [d for d in detections if d["class_id"] == PERSON_CLASS]
                ppe_detections = [d for d in detections if d["class_id"] != PERSON_CLASS]
                tracked_persons = tracker.update([p["box"] for p in persons_raw])

                frame_has_violation, frame_violation_type, frame_violation_conf = False, None, 0.0

                for (person_id, box), person in zip(tracked_persons, persons_raw):
                    associated = associate_ppe_to_person(box, ppe_detections)
                    compliance = determine_compliance(associated)

                    draw_box(frame, box, f"person {person['confidence']:.2f}")
                    draw_person_status(frame, box, person_id, compliance)

                    if compliance["compliance"] == "VIOLATION":
                        frame_has_violation = True
                        if associated["no-helmet"]:
                            frame_violation_type, frame_violation_conf = "No Helmet", associated["no-helmet"]["confidence"]
                        elif associated["no-vest"]:
                            frame_violation_type, frame_violation_conf = "No Vest", associated["no-vest"]["confidence"]

                for det in ppe_detections:
                    draw_box(frame, det["box"], f"{det['class_name']} {det['confidence']:.2f}", color=(200, 200, 0))

                event_log.update(frame_number, fps, frame_has_violation, frame_violation_type, frame_violation_conf)

                if event_log.active:
                    cv2.putText(frame, f"VIOLATION: {event_log.violation_type}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                elapsed = time.time() - start_time
                current_fps = frame_number / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {current_fps:.1f}  Frame: {frame_number}", (20, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            except Exception as frame_error:
                # One bad frame (e.g. a transient inference error) shouldn't kill the whole run.
                skipped_frames += 1
                print(f"[frame {frame_number}] skipped due to error: {frame_error}")

            writer.write(frame)

    except Exception as loop_error:
        print("run_ppe_pipeline stopped early due to an unexpected error:")
        traceback.print_exc()

    finally:
        event_log.close_if_active(frame_number, fps)
        cap.release()
        writer.release()

    elapsed = time.time() - start_time
    print(f"\nFrames processed: {frame_number}  |  skipped: {skipped_frames}  |  "
          f"time: {elapsed:.1f}s  |  avg FPS: {frame_number / elapsed:.1f}" if elapsed > 0 else "")
    print(f"Violation events: {len(event_log.events)}")
    for event in event_log.events:
        print(f"  Event {event['event_id']}: {event['violation']} | {event['start_time']} -> {event['end_time']} | conf={event['confidence']:.2f}")
    return event_log.events


# Outer safety net: even a totally unexpected exception here won't crash the kernel.
try:
    events = run_ppe_pipeline(INPUT_VIDEO, OUTPUT_VIDEO, FINAL_MODEL)
except Exception as e:
    print(f"run_ppe_pipeline failed entirely: {e}")
    traceback.print_exc()
    events = []

print(f"\nCell finished. events variable is set ({len(events)} event(s)) — safe to continue to next cell.")


# In[21]:


import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# Only define these if they don't already exist in this session
if "INPUT_VIDEO" not in globals():
    INPUT_VIDEO = Path("/kaggle/working/input_video.mp4")

if "OUTPUT_DIR" not in globals():
    OUTPUT_DIR = Path("/kaggle/working/opencv_output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if "OUTPUT_VIDEO" not in globals():
    OUTPUT_VIDEO = OUTPUT_DIR / "ppe_compliance_output.mp4"

if "IMG_SIZE" not in globals():
    IMG_SIZE = 640

if "CONF_THRESHOLD" not in globals():
    CONF_THRESHOLD = 0.25

if "IOU_THRESHOLD" not in globals():
    IOU_THRESHOLD = 0.45

if "DEVICE" not in globals():
    DEVICE = 0  # GPU index, or "cpu"

# FINAL_MODEL can't be conjured out of nowhere — it's your trained YOLO model.
# This just gives a clear error instead of a cryptic NameError further down.
if "FINAL_MODEL" not in globals():
    raise NameError(
        "FINAL_MODEL is not defined in this session. You need to reload it, e.g.:\n"
        "    from ultralytics import YOLO\n"
        "    FINAL_MODEL = YOLO('/path/to/your/best_or_epochN.pt')\n"
        "before running the cells below."
    )

print("Config ready:")
print(f"  INPUT_VIDEO   = {INPUT_VIDEO}")
print(f"  OUTPUT_VIDEO  = {OUTPUT_VIDEO}")
print(f"  IMG_SIZE={IMG_SIZE}  CONF_THRESHOLD={CONF_THRESHOLD}  IOU_THRESHOLD={IOU_THRESHOLD}  DEVICE={DEVICE}")


# In[22]:


from IPython.display import Video, display

MAX_EMBED_MB = 50  # embedding as base64 can crash the browser/kernel on large files

try:
    if not OUTPUT_VIDEO.exists():
        print(f"Output video was not created: {OUTPUT_VIDEO}")
    else:
        size_mb = OUTPUT_VIDEO.stat().st_size / (1024 * 1024)
        print(f"Output: {OUTPUT_VIDEO}  ({size_mb:.2f} MB)")

        if size_mb == 0:
            print("Warning: output file exists but is 0 bytes — likely the writer failed silently. Not displaying.")
        elif size_mb > MAX_EMBED_MB:
            print(f"Video is {size_mb:.1f} MB, over the {MAX_EMBED_MB} MB embed limit — "
                  f"skipping inline display to avoid crashing the browser/kernel.")
            print(f"Open/download it directly instead: {OUTPUT_VIDEO}")
        else:
            try:
                display(Video(str(OUTPUT_VIDEO), embed=True))
            except Exception as display_error:
                print(f"Could not embed video inline: {display_error}")
                print(f"Open/download it directly instead: {OUTPUT_VIDEO}")

except Exception as e:
    print(f"Unexpected error while checking/displaying output video: {e}")

print("\nCell finished — safe to continue.")


# ### Optional: raw throughput benchmark
# 
# Measures YOLO inference latency alone (no drawing/tracking overhead) to know the model's true
# FPS ceiling for a deployment target.
# 

# In[23]:


if "open_video_safe" not in globals():
    def open_video_safe(path):
        path = str(path)
        if not os.path.exists(path):
            print(f"[open_video_safe] File does not exist: {path}")
            return None
        if os.path.getsize(path) == 0:
            print(f"[open_video_safe] File exists but is 0 bytes: {path}")
            return None
        for backend, label in [(cv2.CAP_ANY, "default"), (cv2.CAP_FFMPEG, "CAP_FFMPEG")]:
            cap = cv2.VideoCapture(path, backend)
            if cap.isOpened():
                print(f"[open_video_safe] Opened successfully with backend: {label}")
                return cap
            cap.release()
        print(f"[open_video_safe] Could not open with any backend: {path}")
        return None

cap = open_video_safe(INPUT_VIDEO)
latencies, frame_count, skipped_frames = [], 0, 0

if cap is None:
    print("Aborting benchmark: no usable video source.")
else:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            try:
                t0 = time.perf_counter()
                FINAL_MODEL.predict(source=frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)
                latencies.append(time.perf_counter() - t0)
            except Exception as frame_error:
                skipped_frames += 1
                print(f"[frame {frame_count}] inference skipped due to error: {frame_error}")
    except Exception as loop_error:
        print("Benchmark loop stopped early due to an unexpected error:")
        traceback.print_exc()
    finally:
        cap.release()

    if latencies:
        avg_latency = np.mean(latencies)
        print(f"\nFrames: {frame_count}  |  skipped: {skipped_frames}  |  "
              f"avg inference latency: {avg_latency * 1000:.1f} ms  |  ~{1 / avg_latency:.1f} FPS")
    else:
        print(f"\nFrames read: {frame_count}, but no successful inference calls — can't compute latency.")

print("\nCell finished — safe to continue.")

