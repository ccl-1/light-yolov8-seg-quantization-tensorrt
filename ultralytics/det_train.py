from ultralytics.engine.model import YOLO

# Load a model
model = YOLO('cfg/models/v8/yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='cfg/datasets/railDet19.yaml', epochs=100, imgsz=640)

