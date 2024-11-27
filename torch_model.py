import torchvision
import torch
import torchinfo
class FasterRCNN(torch.nn.Module):
    def __init__(self,num_classes:int)->None:
        super(FasterRCNN, self).__init__()
        self.weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.model=torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=self.weights)
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor=torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_channels=self.in_features,num_classes=num_classes)
    def forward(self,x,target=None)->torch.Tensor:
        x=self.model(x,target)
        return x

device=torch.device('cuda:0')
model=FasterRCNN(num_classes=2)

model.to(device)
input_size=torch.rand((2,3,1080,1920),dtype=torch.float32).to(device)
model.eval()
pred=model(input_size)
torchinfo.summary(model=model,input_size=(2,3,1080,1920))