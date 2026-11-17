import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from GTSDBTEST import GTSDB, collate, ROOT, NUM_SIGN_CLASSES  # your training file

HERE = Path(__file__).parent
WEIGHTS_PATH = HERE / "fasterrcnn_gtsdb.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Same hold-out split as training (same seed and fraction)
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

metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

with torch.no_grad():
    for imgs, targets in loader:
        outs = model([im.to(device) for im in imgs])
        preds = [{k: v.cpu() for k, v in o.items()} for o in outs]
        gts = [{k: v.cpu() for k, v in t.items()} for t in targets]
        metric.update(preds, gts)

res = metric.compute()
print(f"\nmAP (IoU 0.50:0.95): {res['map'].item():.3f}")
print(f"AP50 (IoU 0.50):     {res['map_50'].item():.3f}")
print(f"AP75 (IoU 0.75):     {res['map_75'].item():.3f}")
print(f"AP small / medium / large: "
      f"{res['map_small'].item():.3f} / {res['map_medium'].item():.3f} / {res['map_large'].item():.3f}")

print("\nAP per class (class id: AP):")
for c, ap in zip(res["classes"].tolist(), res["map_per_class"].tolist()):
    print(f"  {c - 1:2d}: {ap:.3f}")  # -1 converts back to GTSDB ids 0..42