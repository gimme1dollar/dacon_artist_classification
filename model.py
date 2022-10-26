import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, num_classes=50):
        super(BaseModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
