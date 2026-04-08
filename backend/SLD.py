from imports import *


class SLD(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        backbone = models.resnet50(weights="DEFAULT")

        # Remove the final FC layer — keep everything up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        in_features = 2048

        # Classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def freeze_backbone(self):
        """Freeze all backbone parameters (useful for warm-up phase)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for full fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True
