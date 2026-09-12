import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def LesionClassifier(num_classes=3):
    # Use a lightweight pretrained model
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # Replace classifier head for our num_classes
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
