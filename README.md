## Lightweight Railway Track Segmentation
<div align=center>
<img src="https://github.com/ccl-1/light-yolov8-seg-quantization-tensorrt/blob/main/doc/result.gif" > 
</div>


This repository is designed for lightweight railway track segmentation, enabling real-time performance on resource-constrained edge devices (e.g., Jetson Nano).

The inference acceleration part is located in the `main` branch, while the lightweight model designs are in the `light_version` branch.

Our algorithm is primarily based on these following reference, with optimizations and modifications made on top of them. Installation, configuration, etc., are the same as them.

### Reference Links:
- https://github.com/ultralytics/ultralytics
- https://github.com/CYYAI/AiInfer

#### Note:
When utilizing our algorithm for acceleration, it's essential to generate the TRT (TensorRT) file using your own computer. This file requires some local device information, and generating it on other devices may lead to incompatibility.

### Model Inference Results:

Here are the inference results of several models, along with corresponding images:

<img src="https://github.com/ccl-1/light-yolov8-seg-quantization-tensorrt/blob/main/doc/compare.png" width="500px"><img src="https://github.com/ccl-1/light-yolov8-seg-quantization-tensorrt/blob/main/doc/val_batch1_pred.jpg" width="400px">


Feel free to explore the repository and utilize the models for your railway track segmentation tasks.
