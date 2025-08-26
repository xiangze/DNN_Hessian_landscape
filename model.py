from torch import nn
from torchvision import  models

# ----------------------------
# 2) Model (ResNet18 for CIFAR-10)
# ----------------------------
def make_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model
