import csv
import time
import numpy as np
import torch
from collections import defaultdict
from os import path
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ROOT = r"C:\Users\Meier\Desktop\GTSDB\Train"
NUM_SIGN_CLASSES = 43  # GTSDB class ids 0..42

HERE = Path(__file__).parent
WEIGHTS_PATH = HERE / "fasterrcnn_gtsdb.pth"
RESULTS_PATH = HERE / "results.csv"

NUM_EPOCHS = 100
EVAL_EVERY = 1     # evaluate on the validation set every N epochs (metric columns are blank otherwise)
SCORE_THR = 0.5    # confidence needed for a detection to count for precision / recall
IOU_THR = 0.5      # box overlap needed to match a detection to a real sign
BG = NUM_SIGN_CLASSES
LOSS_KEYS = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]


class GTSDB(Dataset):
    def __init__(self, root):
        self.root = root
        self.boxes = defaultdict(list)   # filename -> [[x1,y1,x2,y2,cls], ...]

        with open(path.join(root, "gt.txt")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                name, x1, y1, x2, y2, c = line.split(";")
                self.boxes[name].append([float(x1), float(y1), float(x2), float(y2), int(c)])

        # every image in the folder, including ones with no signs
        from os import listdir
        self.files = sorted(f for f in listdir(root) if f.lower().endswith(".ppm"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(path.join(self.root, name)).convert("RGB")

        entries = self.boxes.get(name, [])
        boxes = torch.tensor([e[:4] for e in entries], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([e[4] + 1 for e in entries], dtype=torch.int64)  # 0 = background

        return TF.to_tensor(img), {"boxes": boxes, "labels": labels}


def collate(batch):
    return tuple(zip(*batch))


def build_model():
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", min_size=1000, max_size=1700)
    inF = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(inF, NUM_SIGN_CLASSES + 1)
    return model


def update_cm(cm, pred, tgt):
    """cm[predicted, true]; index BG = background (missed sign / false alarm)."""
    keep = pred["scores"] >= SCORE_THR
    pb = pred["boxes"][keep]
    pl = (pred["labels"][keep] - 1).tolist()
    ps = pred["scores"][keep]
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
                cm[pl[p], gl[best]] += 1
            else:
                cm[pl[p], BG] += 1
    else:
        for p in pl:
            cm[p, BG] += 1
    for g in range(len(gl)):
        if g not in matched:
            cm[BG, gl[g]] += 1


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (validation losses, precision, recall, AP50, mAP50-95)."""
    from torchmetrics.detection import MeanAveragePrecision

    # 1) validation losses. torchvision only returns losses in train mode; this model
    #    uses frozen BatchNorm and has no dropout, so that is safe (no weights change).
    model.train()
    sums, nb = defaultdict(float), 0
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        tg = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for k, v in model(imgs, tg).items():
            sums[k] += v.item()
        nb += 1
    vloss = {k: v / nb for k, v in sums.items()}

    # 2) predictions -> AP, precision, recall
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    cm = np.zeros((BG + 1, BG + 1), dtype=int)
    for imgs, targets in loader:
        outs = model([im.to(device) for im in imgs])
        preds = [{k: v.cpu() for k, v in o.items()} for o in outs]
        gts = [{k: v.cpu() for k, v in t.items()} for t in targets]
        metric.update(preds, gts)
        for p, g in zip(preds, gts):
            update_cm(cm, p, g)
    res = metric.compute()

    tp = np.trace(cm[:BG, :BG])
    n_pred = cm[:BG, :].sum()      # detections above SCORE_THR
    n_true = cm[:, :BG].sum()      # real signs
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    return vloss, precision, recall, res["map_50"].item(), res["map"].item()


FIELDS = [
    "epoch", "time", "epoch_time",
    "train/loss_classifier", "train/loss_box_reg", "train/loss_objectness",
    "train/loss_rpn_box_reg", "train/total_loss",
    "metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95",
    "val/loss_classifier", "val/loss_box_reg", "val/loss_objectness",
    "val/loss_rpn_box_reg", "val/total_loss",
    "lr",
]


def write_csv(rows):
    with open(RESULTS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    full = GTSDB(ROOT)
    print("images found:", len(full), "| images with signs:", len(full.boxes), flush=True)

    nVal = int(len(full) * 0.15)
    train, val = random_split(full, [len(full) - nVal, nVal],
                              generator=torch.Generator().manual_seed(0))
    loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val, batch_size=2, shuffle=False, collate_fn=collate, num_workers=0)

    print("building model (first run downloads weights)...", flush=True)
    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)

    rows = []
    train_start = time.time()

    try:
        for epoch in range(NUM_EPOCHS):
            model.train()
            t0 = time.time()
            sums = {k: torch.zeros((), device=device) for k in LOSS_KEYS}
            n = 0

            for i, (imgs, targets) in enumerate(loader):
                imgs = [im.to(device) for im in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
                opt.zero_grad()
                loss.backward()
                opt.step()
                for k in LOSS_KEYS:
                    sums[k] += loss_dict[k].detach()
                n += 1
                if i % 10 == 0:
                    print(f"epoch {epoch + 1} batch {i}/{len(loader)} loss {loss.item():.3f}", flush=True)

            tr = {k: v.item() / n for k, v in sums.items()}   # .item() syncs the GPU
            epoch_time = time.time() - t0

            row = {f: "" for f in FIELDS}
            row["epoch"] = epoch + 1
            row["epoch_time"] = round(epoch_time, 3)
            for k in LOSS_KEYS:
                row[f"train/{k}"] = round(tr[k], 5)
            row["train/total_loss"] = round(sum(tr.values()), 5)
            row["lr"] = opt.param_groups[0]["lr"]

            if EVAL_EVERY and (epoch + 1) % EVAL_EVERY == 0:
                vloss, prec, rec, ap50, ap = evaluate(model, val_loader, device)
                row["metrics/precision"] = round(prec, 5)
                row["metrics/recall"] = round(rec, 5)
                row["metrics/mAP50"] = round(ap50, 5)
                row["metrics/mAP50-95"] = round(ap, 5)
                for k in LOSS_KEYS:
                    row[f"val/{k}"] = round(vloss[k], 5)
                row["val/total_loss"] = round(sum(vloss.values()), 5)

            row["time"] = round(time.time() - train_start, 3)   # cumulative, includes evaluation
            rows.append(row)
            write_csv(rows)                                     # CSV + weights after every epoch
            torch.save(model.state_dict(), WEIGHTS_PATH)

            msg = (f"epoch {epoch + 1}/{NUM_EPOCHS}: train loss {row['train/total_loss']:.3f} | "
                   f"epoch time {epoch_time / 60:.1f} min ({epoch_time:.0f} s) | "
                   f"total {row['time'] / 60:.1f} min")
            if row["metrics/mAP50"] != "":
                msg += (f" | P {row['metrics/precision']:.3f} R {row['metrics/recall']:.3f} "
                        f"mAP50 {row['metrics/mAP50']:.3f} mAP50-95 {row['metrics/mAP50-95']:.3f}")
            print(msg, flush=True)

    except KeyboardInterrupt:
        print("\nStopped by user, saving...", flush=True)

    finally:
        torch.save(model.state_dict(), WEIGHTS_PATH)
        write_csv(rows)
        if rows:
            times = [r["epoch_time"] for r in rows]
            print("\n=== TIME PER EPOCH ===")
            for r in rows:
                print(f"epoch {r['epoch']}: {r['epoch_time'] / 60:.1f} min ({r['epoch_time']:.0f} s)")
            print(f"average: {sum(times) / len(times) / 60:.1f} min | "
                  f"total: {(time.time() - train_start) / 60:.1f} min")
        print(f"saved {WEIGHTS_PATH} and {RESULTS_PATH}")