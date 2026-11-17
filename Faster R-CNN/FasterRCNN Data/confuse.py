import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import random_split
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from GTSDBTEST import GTSDB, ROOT, NUM_SIGN_CLASSES  # your training file

HERE = Path(__file__).parent
WEIGHTS_PATH = HERE / "fasterrcnn_gtsdb.pth"
SCORE_THR = 0.25   # minimum confidence for a detection
IOU_THR = 0.5      # box overlap needed to match a detection to a real sign
ALPHABETICAL = True   # True = sorted by name like your example, False = class id order
ANNOTATE = False      # True = also write the raw counts inside the cells

# [] = the whole validation split. Or list Train files, e.g. ["00012.ppm", "00040.ppm"]
IMAGE_NAMES = []

NAMES = [
    "speed limit 20", "speed limit 30", "speed limit 50", "speed limit 60",
    "speed limit 70", "speed limit 80", "end speed limit 80", "speed limit 100",
    "speed limit 120", "no overtaking", "no overtaking (trucks)",
    "priority at next intersection", "priority road", "yield", "stop",
    "no traffic both ways", "no trucks", "no entry", "danger", "bend left",
    "bend right", "bend", "uneven road", "slippery road", "road narrows",
    "construction", "traffic signal", "pedestrian crossing", "school crossing",
    "cycles crossing", "snow", "animals", "end of all restrictions", "go right",
    "go left", "go straight", "go right or straight", "go left or straight",
    "keep right", "keep left", "roundabout", "end no overtaking",
    "end no overtaking (trucks)",
]
BG = NUM_SIGN_CLASSES   # index of the "background" row/column

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

full = GTSDB(ROOT)
if IMAGE_NAMES:
    indices = [full.files.index(n) for n in IMAGE_NAMES]
else:
    nVal = int(len(full) * 0.15)
    _, val = random_split(full, [len(full) - nVal, nVal],
                          generator=torch.Generator().manual_seed(0))
    indices = list(val.indices)
print("images tested:", len(indices))

model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None,
                                   min_size=1000, max_size=1700)
inF = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(inF, NUM_SIGN_CLASSES + 1)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device).eval()

# cm[predicted, true]; last index = background
cm = np.zeros((BG + 1, BG + 1), dtype=int)

with torch.no_grad():
    for idx in indices:
        img, tgt = full[idx]
        out = model([img.to(device)])[0]

        keep = out["scores"] >= SCORE_THR
        pb = out["boxes"][keep].cpu()
        pl = (out["labels"][keep].cpu() - 1).tolist()
        ps = out["scores"][keep].cpu()
        gb = tgt["boxes"]
        gl = (tgt["labels"] - 1).tolist()

        matched = set()
        if len(pb) and len(gb):
            iou = box_iou(pb, gb)
            for p in ps.argsort(descending=True).tolist():
                row = iou[p].clone()
                for g in matched:
                    row[g] = -1
                best = int(row.argmax())
                if row[best] >= IOU_THR:
                    matched.add(best)
                    cm[pl[p], gl[best]] += 1     # correct or confused class
                else:
                    cm[pl[p], BG] += 1           # false alarm (predicted a sign, none there)
        else:
            for p in pl:
                cm[p, BG] += 1
        for g in range(len(gl)):
            if g not in matched:
                cm[BG, gl[g]] += 1               # missed sign (predicted background)

# Reorder and normalize each column (each true class sums to 1)
order = sorted(range(NUM_SIGN_CLASSES), key=lambda i: NAMES[i]) if ALPHABETICAL \
        else list(range(NUM_SIGN_CLASSES))
order = order + [BG]
labels = [NAMES[i] for i in order[:-1]] + ["background"]
cm_o = cm[np.ix_(order, order)]
norm = cm_o / np.maximum(cm_o.sum(axis=0, keepdims=True), 1)
shown = np.ma.masked_less(norm, 0.005)   # hide near-empty cells like your example

cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("white")

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=1)
n = len(labels)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=90, fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("True")
ax.set_ylabel("Predicted")
ax.set_title("Confusion Matrix Normalized")
if ANNOTATE:
    for i in range(n):
        for j in range(n):
            if cm_o[i, j] > 0:
                ax.text(j, i, cm_o[i, j], ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
out_png = HERE / "confusion_matrix_normalized.png"
plt.savefig(out_png, dpi=200)
np.savetxt(HERE / "confusion_matrix_counts.csv", cm_o, fmt="%d", delimiter=",")
print("saved", out_png)
plt.show()