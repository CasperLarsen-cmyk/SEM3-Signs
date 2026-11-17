import sys
import torch
from pathlib import Path
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

HERE = Path(__file__).parent
IMAGE_PATH = HERE / "test.jpg"                # image to test
WEIGHTS_PATH = HERE / "fasterrcnn_gtsdb.pth"  # trained weights
THRESHOLD = 0.5                               # minimum confidence to show

# Optional: python predict.py other.jpg other_weights.pth
if len(sys.argv) > 1:
    IMAGE_PATH = Path(sys.argv[1])
if len(sys.argv) > 2:
    WEIGHTS_PATH = Path(sys.argv[2])

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
NUM_CLASSES = len(NAMES) + 1  # +1 for background

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Must match the architecture used in training
model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None,
                                   min_size=1000, max_size=1700)
inF = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(inF, NUM_CLASSES)

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device).eval()

img = Image.open(IMAGE_PATH).convert("RGB")
with torch.no_grad():
    out = model([TF.to_tensor(img).to(device)])[0]

draw = ImageDraw.Draw(img)
count = 0
for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
    if score < THRESHOLD:
        continue
    count += 1
    x1, y1, x2, y2 = box.tolist()
    text = f"{NAMES[label.item() - 1]} {score:.2f}"
    print(text, [round(v) for v in (x1, y1, x2, y2)])
    draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
    tb = draw.textbbox((x1, y1 - 14), text)
    draw.rectangle(tb, fill="white")
    draw.text((x1, y1 - 14), text, fill="black")

print(f"{count} detections above {THRESHOLD}")
result = HERE / "test_result.jpg"
img.save(result)
print("saved", result)
img.show()