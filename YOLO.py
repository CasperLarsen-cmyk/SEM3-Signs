from ultralytics import YOLO

ModelPath = "runs/detect/train-14/weights/best.pt"
model = YOLO(ModelPath)         # load a custom model

#model = YOLO("yolov8n.yaml")    # build a new model from YAML
#model = YOLO("yolov8n.pt")      # load pretrained

print(model)
resultTrain = model.train(data="coco.yaml", epochs=25, imgsz=640, device=0)#, batch=32
model.export(format="onnx")

results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
#print(results.boxes)
for result in results:
    for box in result.boxes:
        print(box.cls.data, box.conf.data, box.xyxy.data)