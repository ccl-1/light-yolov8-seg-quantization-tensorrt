from ultralytics.engine.model import YOLO

# 32 倍数 640=32*20,  800=32*25960=32*30, 1024=32*32


# model = YOLO('cfg/models/v8/yolov8n-seg.yaml').load('yolov8n-seg.pt') 
# model.train(data='cfg/datasets/railseg19_semantic.yaml', epochs=300, batch=4, imgsz=[1024,576]) #[w,h] 


# model = YOLO('cfg/models/v8/yolov8n-pconv.yaml', task='segment').load('yolov8n-seg.pt') 
# model = YOLO('cfg/models/v8/yolov8n-c3faster.yaml', task='segment').load('yolov8n-seg.pt') 
# model = YOLO('cfg/models/v8/yolov8n-c2ffaster.yaml', task='segment').load('runs/segment/yolov8n-seg/weights/best.pt') 
# model = YOLO('cfg/models/v8/yolov8n-sharedHead.yaml', task='segment').load('runs/segment/sharedHead/weights/best.pt') 

# model = YOLO('cfg/models/v8/yolov8n-efficient.yaml', task='segment').load('yolov8n-seg.pt') 
# model = YOLO('cfg/models/v8/yolov8-c2ffaster_ema.yaml', task='segment').load('yolov8n-seg.pt') 
# model = YOLO('cfg/models/v8/yolov8n-lighthead.yaml', task='segment').load('yolov8n-seg.pt') 
# model = YOLO('cfg/models/v8/yolov8-fasternet.yaml', task='segment').load('yolov8n-seg.pt') 
# model = YOLO('cfg/models/v8/yolov8-light.yaml', task='segment').load('yolov8n-seg.pt') 
model = YOLO('cfg/models/v8/yolov8-light2.yaml', task='segment').load('yolov8n-seg.pt') 
# model = YOLO('cfg/models/v8/yolov8n-seg_c3ghost.yaml', task='segment').load('yolov8n-seg.pt') 


# 目前得是 lightGroup_c3ghost
# TODO group head + C3Ghost + C2F  light2  Group_c3ghost_c2f
# TODO light head + C3Ghost + C2F          light_c3ghost_c2f
# TODO 最后测试一组 2层输出地 ... 

# model.train(data='cfg/datasets/railseg19_semantic.yaml', task='segment', epochs=100, batch=4, imgsz=[1024,576]) #[w,h] 
# model.train(data='cfg/datasets/railseg19-seg.yaml', task='segment', epochs=100, batch=8, imgsz=800) #[w,h]    25         480, 800
model.train(data='cfg/datasets/railseg19-seg.yaml', task='segment', epochs=100, batch=8, imgsz=640) #[w,h]      20  (1, 3, 384, 640)  train4
# model.train(data='cfg/datasets/railseg19-seg.yaml', task='segment', epochs=100, batch=8, imgsz=480) #[w,h]    15         288, 480
# model.train(data='cfg/datasets/railseg19-seg.yaml', task='segment', epochs=100, batch=8, imgsz=320) #[w,h]    10


# model.train(data='cfg/datasets/railseg19-seg.yaml', task='segment', epochs=150, batch=16, imgsz=800, optimizer='Lion',) #[w,h] 
