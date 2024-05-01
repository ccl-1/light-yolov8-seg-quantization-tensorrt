from ultralytics import YOLO



# 修改前后 模型 ... 
# model_path = "/media/ubuntu/zoro/ubuntu/code/ultralytics/ultralytics/runs/segment/yolov8n-seg/weights/best.pt"
model_path = "/media/ubuntu/zoro/ubuntu/code/ultralytics/ultralytics/runs/segment/yolov8n-seg-ours/weights/best.pt"
# model_path = "/media/ubuntu/zoro/ubuntu/code/ultralytics/ultralytics/runs/segment/train4/weights/best.pt"
# model_path = "./yolov8-optimization/models/trt/optimized_fp16_480_800.engine"

# Load a model
model = YOLO(model_path)  # load a custom model

import time
s = time.time()



# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps   # a list contains map50-95(B) of each category
metrics.seg.map    # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps   # a list contains map50-95(M) of each category


e = time.time()
fps = 1.0 / (e-s) * 1275
print(fps)
# 101.14106258173906

# 修改前后 模型推理速度对比 ... 
# Speed: 0.1ms preprocess, 2.4ms inference, 0.0ms loss, 0.2ms postprocess per image
# Speed: 0.1ms preprocess, 2.0ms inference, 0.0ms loss, 0.3ms postprocess per image

