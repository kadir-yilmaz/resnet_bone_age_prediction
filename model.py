import torch
import torch.nn as nn
import torchvision.models as models


class BoneAgeModel(nn.Module):
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # ResNet50 backbone
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Gender embedding
        self.gender_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Regression head (2048 + 32 -> 1)
        self.regressor = nn.Sequential(
            nn.Linear(num_features + 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, image, gender):
        img_features = self.backbone(image)
        gender_features = self.gender_embedding(gender.float())
        combined = torch.cat([img_features, gender_features], dim=1)
        return self.regressor(combined)


def get_model(pretrained=True):
    return BoneAgeModel(pretrained=pretrained)
