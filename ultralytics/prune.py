from ultralytics import YOLO
import torch
from ultralytics.nn.modules import Bottleneck, Conv, C2f, SPPF, Detect

# Load a model
yolo = YOLO("last.pt")  # build a new model from scratch
model = yolo.model

ws = []
bs = []

for name, m in model.named_modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        w = m.weight.abs().detach()
        b = m.bias.abs().detach()
        ws.append(w)
        bs.append(b)
        # print(name, w.max().item(), w.min().item(), b.max().item(), b.min().item())
# keep
factor = 0.8
ws = torch.cat(ws)
threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]
print(threshold)

def prune_conv(conv1: Conv, conv2: Conv):
    gamma = conv1.bn.weight.data.detach()
    beta = conv1.bn.bias.data.detach()
    keep_idxs = []
    local_threshold = threshold
    while len(keep_idxs) < 8:
        keep_idxs = torch.where(gamma.abs() >= local_threshold)[0]
        local_threshold = local_threshold * 0.5
    n = len(keep_idxs)
    # n = max(int(len(idxs) * 0.8), p)
    # print(n / len(gamma) * 100)
    # scale = len(idxs) / n
    conv1.bn.weight.data = gamma[keep_idxs]
    conv1.bn.bias.data = beta[keep_idxs]
    conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
    conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
    conv1.bn.num_features = n
    conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs]
    conv1.conv.out_channels = n

    if conv1.conv.bias is not None:
        conv1.conv.bias.data = conv1.conv.bias.data[keep_idxs]

    if not isinstance(conv2, list):
        conv2 = [conv2]

    for item in conv2:
        if item is not None:
            if isinstance(item, Conv):
                conv = item.conv
            else:
                conv = item
            conv.in_channels = n
            conv.weight.data = conv.weight.data[:, keep_idxs]


def prune(m1, m2):
    if isinstance(m1, C2f):  # C2f as a top conv
        m1 = m1.cv2

    if not isinstance(m2, list):  # m2 is just one module
        m2 = [m2]

    for i, item in enumerate(m2):
        if isinstance(item, C2f) or isinstance(item, SPPF):
            m2[i] = item.cv1

    prune_conv(m1, m2)


for name, m in model.named_modules():
    if isinstance(m, Bottleneck):
        prune_conv(m.cv1, m.cv2)

seq = model.model
for i in range(3, 9):
    if i in [6, 4, 9]: continue
    prune(seq[i], seq[i + 1])

detect: Detect = seq[-1]
last_inputs = [seq[15], seq[18], seq[21]]
colasts = [seq[16], seq[19], None]
for last_input, colast, cv2, cv3 in zip(last_inputs, colasts, detect.cv2, detect.cv3):
    prune(last_input, [colast, cv2[0], cv3[0]])
    prune(cv2[0], cv2[1])
    prune(cv2[1], cv2[2])
    prune(cv3[0], cv3[1])
    prune(cv3[1], cv3[2])

for name, p in yolo.model.named_parameters():
    p.requires_grad = True

# yolo.val() # 剪枝模型进行验证 yolo.val(workers=0)
# yolo.export(format="onnx") # 导出为onnx文件
# yolo.train(data="VOC.yaml", epochs=100) # 剪枝后直接训练微调

torch.save(yolo.ckpt, "prune.pt")

print("done")


"""
们通过上述代码可以完成剪枝工作并将剪枝好的模型进行保存，用于finetune，有以下几点说明：
在本次剪枝中我们利用factor变量来控制剪枝的保留率

我们用来剪枝的模型一定是约束训练的模型，即对BN层加上L1正则化后训练的模型

约束训练后的b.min().item值非常小，接近于0或者等于0，可以依据此来判断加载的模型是否正确

我们可以选择将yolo.train()取消注释，在剪枝完成直接进入微调训练，博主在这里选择先保存剪枝模型

我们可以选择yolo.export()取消注释，将剪枝完成后的模型导出为ONNX，查看对应的大小和channels是否发生改变，以此确认我们完成了剪枝

yolo.val()用于进行模型验证，建议取消注释进行相关验证，之前梁老师说yolo.val()验证的mAP值完全等于0是不正常的，需要检查下剪枝过程是否存在错误，
    最好是有一个值，哪怕非常小，博主剪枝后进行验证的结果如下图所示，可以看到mAP值真的是惨不忍睹(🤣)，所以后续需要finetune模型来恢复我们的精度
"""
