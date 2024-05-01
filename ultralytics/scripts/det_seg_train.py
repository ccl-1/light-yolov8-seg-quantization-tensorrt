from ultralytics.engine.model import YOLO

# Load a model
model = YOLO('cfg/models/v8/yolov8n-seg.yaml').load('yolov8n.pt') 

# Train the model
model.train(data='cfg/datasets/railseg19.yaml', epochs=100, imgsz=640)






