from ultralytics.engine.model import YOLO
from thop import profile
from ultralytics.nn.tasks import DetectionModel, SegmentationModel


class YOLOV8(YOLO):
    def __init__(self, yaml="", weight="", task=None) -> None:
        super().__init__(yaml, task)
        if weight:
            self.load(weight)


if __name__ == '__main__':

    #  ------------  baseline ----------------------------------------
    # yaml = 'cfg/models/v8/yolov8n.yaml'               # 9.8 - 7.28
    # yaml = 'cfg/models/v8/yolov8n-seg.yaml'           # 12.6 理论， 现在50多，代码出问题了？ 要不
    # ---------------------------------------------------------------- 

    # yaml = 'cfg/models/v8/yolov8n-pconv.yaml'         # 64.2
    # yaml = 'cfg/models/v8/yolov8n-c3faster.yaml'      # 53.4
    # yaml = 'cfg/models/v8/yolov8n-c2ffaster.yaml'     # 53.7
    yaml = 'cfg/models/v8/yolov8n-light2.yaml'      # 50.2
    # yaml = 'cfg/models/v8/yolov8n-seg.yaml'      # 50.2


    model_path = "/media/ubuntu/zoro/ubuntu/code/ultralytics/ultralytics/runs/segment/yolov8n-seg-ours/weights/best.pt"
    # model_path = "/media/ubuntu/zoro/ubuntu/code/ultralytics/ultralytics/runs/segment/yolov8n-seg/weights/best.pt"

    model = YOLOV8(yaml=yaml,weight=model_path,task='segment') 
    # model.info(detailed=True, verbose=True)
    # model.profile((480,800))
    model.profile(640)

"""
------------org-model--------------------
input size:  torch.Size([1, 3, 640, 640])
output size
torch.Size([1, 144, 80, 80])
torch.Size([1, 144, 40, 40])
torch.Size([1, 144, 20, 20])
-----
torch.Size([32, 8400])
-----
torch.Size([32, 160, 160])

--------------ours-model-----------------
input size:  torch.Size([1, 3, 480, 800])
output size
torch.Size([1, 144, 60, 100])
torch.Size([1, 144, 30, 50])
-----
torch.Size([32, 7500])
-----
torch.Size([32, 120, 200])
"""