import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from GTSDBTEST import GTSDB, collate, ROOT, NUM_SIGN_CLASSES  # your training file

HERE = Path(__file__).parent
WEIGHTS_PATH = HERE / "fasterrcnn_gtsdb.pth"
SCORE_THR = 0.5   # ignore detections below this confidence
IOU_THR = 0.5     # box overlap needed to count a detection as matching a sign
BG = NUM_SIGN_CLASSES  # extra row/column for "background / nothing"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Same split as training (same seed and fraction)
full = GTSDB(ROOT)
nVal = int(len(full) * 0.15)
_, val = random_split(full, [len(full) - nVal, nVal],
                      generator=torch.Generator().manual_seed(0))
loader = DataLoader(val, batch_size=2, shuffle=False, collate_fn=collate)

model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None,
                                   min_size=1000, max_size=1700)
inF = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(inF, NUM_SIGN_CLASSES + 1)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device).eval()

# rows = true class, columns = predicted class, last index = background
cm = np.zeros((BG + 1, BG + 1), dtype=int)

with torch.no_grad():
    for imgs, targets in loader:
        outs = model([im.to(device) for im in imgs])
        for out, tgt in zip(outs, targets):
            keep = out["scores"] >= SCORE_THR
            pb = out["boxes"][keep].cpu()
            pl = (out["labels"][keep].cpu() - 1).tolist()
            ps = out["scores"][keep].cpu()
            gb = tgt["boxes"]
            gl = (tgt["labels"] - 1).tolist()

            matched = set()
            if len(pb) and len(gb):
                iou = box_iou(pb, gb)  # predictions x ground truths
                for p in ps.argsort(descending=True).tolist():
                    row = iou[p].clone()
                    for g in matched:
                        row[g] = -1
                    best = int(row.argmax())
                    if row[best] >= IOU_THR:
                        matched.add(best)
                        cm[gl[best], pl[p]] += 1   # correct or confused class
                    else:
                        cm[BG, pl[p]] += 1         # false positive
            else:
                for p in pl:
                    cm[BG, p] += 1                 # false positive
            for g in range(len(gl)):
                if g not in matched:
                    cm[gl[g], BG] += 1             # missed sign

# Summary
total = cm[:BG, :].sum()
correct = np.trace(cm[:BG, :BG])
missed = cm[:BG, BG].sum()
confused = cm[:BG, :BG].sum() - correct
falsepos = cm[BG, :BG].sum()
print(f"signs in val set: {total}")
print(f"correct: {correct} | wrong class: {confused} | missed: {missed} | false positives: {falsepos}")

np.savetxt(HERE / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

# Plot (color = share of each true class, numbers = counts)
rows = cm.sum(axis=1, keepdims=True)
norm = np.divide(cm, rows, out=np.zeros(cm.shape), where=rows > 0)
norm[BG, BG] = 0

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
for i in range(BG + 1):
    for j in range(BG + 1):
        if cm[i, j] > 0 and not (i == BG and j == BG):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
labels = [str(i) for i in range(BG)] + ["bg"]
ax.set_xticks(range(BG + 1))
ax.set_yticks(range(BG + 1))
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title(f"Confusion matrix (val set, score >= {SCORE_THR}, IoU >= {IOU_THR})")
fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class")
plt.tight_layout()
out_png = HERE / "confusion_matrix.png"
plt.savefig(out_png, dpi=150)
print("saved", out_png)
plt.show()