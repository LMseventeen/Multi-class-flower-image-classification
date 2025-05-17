import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def get_modified_resnet18(num_classes=17, dropout_p=0.5):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # 冻结除layer4和fc外的所有层
    for name, param in model.named_parameters():
        if not name.startswith('layer4') and not name.startswith('fc'):
            param.requires_grad = False
    # 替换全连接层，添加Dropout
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(in_features, num_classes)
    )
    return model 