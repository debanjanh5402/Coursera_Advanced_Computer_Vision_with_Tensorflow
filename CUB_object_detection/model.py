import torch 
from torch import nn

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from torch import nn
def feature_extractor():
    #base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    base = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    
    for param in base.parameters():
        param.requires_grad = True

    """for param in base.features[-3:].parameters():
        param.requires_grad = True"""
        
    model = torch.nn.Sequential(
        base.features,
        base.avgpool,
        torch.nn.Flatten()
    )
    return model



class DualHeadMobileNetObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = feature_extractor()
        input_dim = 1280
        #input_dim = 576

        self.shared_fc = nn.Sequential(nn.Linear(input_dim, 1024),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.4),
                                       #nn.Linear(2048, 1024),
                                       #nn.ReLU(inplace=True),
                                       #nn.Dropout(0.4),
                                       nn.Linear(1024, 512),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.4))

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

        # 5. Localization Head
        self.locator = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid()                
        )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_fc(features)
        
        class_out = self.classifier(shared)
        box_out = self.locator(shared)
        
        return class_out, box_out