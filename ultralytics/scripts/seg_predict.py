from ultralytics import YOLO
import cv2
import numpy as np 
import matplotlib as mpl
from matplotlib import pyplot as  plt
from PIL import Image
import os

# 16进制颜色格式颜色转换为RGB格式
def Hex_to_RGB(hex):
    r = int(hex[1:3],16)
    g = int(hex[3:5],16)
    b = int(hex[5:7], 16)
    return (r,g,b)

def get_cls_color():
    color_list = []
    colors = mpl.colors.CSS4_COLORS  # 字典，包含148个元素
    for k, v in colors.items():
        c = Hex_to_RGB(v)
        color_list.append(c)
    return color_list

color_list = get_cls_color()
# model_dir ='runs/segment/yolov8s_1024_pconv/weights/best.pt'
# Speed: 2.2ms preprocess, 12.3ms inference, 1.7ms postprocess per image at shape (1, 3, 576, 1024)
# YOLOv8s-pconv summary (fused): 129 layers, 22671126 parameters, 0 gradients, 77.5 GFLOPs
# YOLOv8n-pconv summary: 167 layers, 5983238 parameters, 5983222 gradients, 20.8 GFLOPs

# YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
# YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs

# model_dir ='runs/segment/yolov8s_1024/weights/best.pt'
# model_dir = "/media/ubuntu/zoro/ubuntu/code/ultralytics/ultralytics/runs/segment/yolov8n-seg-ours/weights/best.pt"
# model_dir = "/media/ubuntu/zoro/ubuntu/code/ultralytics/ultralytics/runs/segment/train4/weights/best.pt"
model_dir = "/media/ubuntu/zoro/ubuntu/code/ultralytics/ultralytics/runs/segment/yolov8n-seg-ours/weights/best.pt"

# Speed: 2.1ms preprocess, 4.5ms inference, 1.7ms postprocess per image at shape (1, 3, 576, 1024)

# model_dir ='yolov8n-seg.pt'

# source_dir = 'scripts/data/t0.mp4'
# source_dir = 'assets/bus.jpg'
# source_dir = "assets/rs00058.jpg"
source_dir = "assets/imgs"
# save_path = "assets/test.png"

img_name = sorted(os.listdir(source_dir))
for i in img_name:
    img_path = os.path.join(source_dir, i)
    save_path = img_path.replace("imgs", "out")

    model = YOLO(model_dir)  
    # results = model(source_dir,stream=True, save=True)  
    # results = model(source_dir,stream=True)  
    results = model(img_path)  
    annotated_frame = results[0].plot(labels=False,boxes=False)
    res = Image.fromarray(annotated_frame[:,:,::-1])
    res.save(save_path)

    plt.imshow(res)
    plt.show()



# for result in results:
#     img = result.orig_img # The original image as a numpy array.
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs

    # img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
    # im_gpu = torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device).permute(
    #     2, 0, 1).flip(0).contiguous() / 255
    # idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
    # annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)



    # probs = result.probs  # Probs object for classification outputs
    # res_plotted = result.plot()
    # cv2.namedWindow("yolov8_result", cv2.WINDOW_NORMAL)
    # cv2.imshow("yolov8_result", res_plotted)
    # cv2.waitKey(0)
    
    # if boxes is not None:
    #     for box in boxes:
    #         xyxy = box.xyxy.reshape(2,2).cpu().numpy()
    #         c = int(box.cls)
    #         pt1 = np.array((xyxy[0][0], xyxy[0][1]), np.int32)
    #         pt2 = np.array((xyxy[1][0], xyxy[1][1]), np.int32)
    #         cv2.rectangle(img,pt1, pt2, color= color_list[c], thickness=3 ) 

    #     for mask in masks:
    #         pts = mask.xy  
    #         pts = np.array(pts, np.int32)  
    #         cv2.fillPoly(img, [pts], color= color_list[c] ) 
        
    # cv2.imshow('mask', img)
    # cv2.waitKey(0)
    




