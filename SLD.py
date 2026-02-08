from imports import *

class SLD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = models.resnet50(pretrained=True)
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        in_features = 2048  # ResNet-50 final layer feature size
        
        self.header = nn.Sequential(
            nn.Flatten(),
            nn.layerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        out = self.header(features)
        return out