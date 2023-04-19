import torch
from torch import nn
from torchvision.transforms.functional import normalize
from ultralytics import YOLO

from model.ultralytics.nn.tasks import SegmentationModel

# class YOLOEncoder(SegmentationModel):
#     def __init__(self,pretrained: bool = False):
#         super().__init__("model/ultralytics/models/v8/yolov8-seg.yaml",3)
#         if pretrained:
#             self.load_state_dict(torch.hub.load_state_dict_from_url(
#                 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt'))

#     def forward_single_frame(self, x):
#         x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
#         x = self.model[0](x)
#         x = self.model[1](x)
#         f1 = x
#         x = self.model[2](x)
#         x = self.model[3](x)
#         f2 = x
#         x = self.model[4](x)
#         x = self.model[5](x)
#         x = self.model[6](x)
#         f3 = x
#         x = self.model[7](x)
#         x = self.model[8](x)
#         x = self.model[9](x)
#         x = self.model[10](x)
#         x = self.model[11](x)
#         x = self.model[12](x)
#         x = self.model[13](x)
#         x = self.model[14](x)
#         x = self.model[15](x)
#         x = self.model[16](x)
#         f4 = x
#         return [f1, f2, f3, f4]
    
#     def forward_time_series(self, x):
#         B, T = x.shape[:2]
#         features = self.forward_single_frame(x.flatten(0, 1))
#         features = [f.unflatten(0, (B, T)) for f in features]
#         return features

#     def forward(self, x):
#         print("x_shape: {} save: {}".format(x.shape,self.save))
#         return x
#         if x.ndim == 5:
#             return self.forward_time_series(x)
#         else:
#             return self.forward_single_frame(x)

class YOLOEncoder(SegmentationModel):
    def __init__(self,pretrained: bool = False):
        super().__init__(cfg="model/ultralytics/models/v8/yolov8n-seg.yaml")
        
        # if pretrained:
        #     self.ckpt_path("checkpoint/yolov8n-seg.pt")

    def forward_single_frame(self, x):
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        y = []  # outputs
        
        for model_layer in self.model:            
            x = model_layer(x)  # run
            y.append(x)  # save output
        # print("OUTPUTS:",y)
        
        return [y[0],y[2],y[4],y[6]]
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, T)) for f in features]
        return features

    def forward(self, x):
        print("x_shape: {} {}".format(x.shape, x.ndim))
        # return x
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
