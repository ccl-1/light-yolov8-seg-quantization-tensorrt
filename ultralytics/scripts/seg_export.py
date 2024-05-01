from ultralytics.engine.model import YOLO


# Load a model
model = YOLO('yolov8n-seg.pt')
#model = YOLO('path/to/best.pt')  


model.export(format='onnx')






