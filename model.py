import torch
import torch.nn as nn
from torchvision.models import resnet50

class AgeGenderModel(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False, dropout: float = 0.5, hidden_dim: int = 512):
        super().__init__()
        
        # 1. Backbone Setup
        if pretrained:
            try:
                from torchvision.models import ResNet50_Weights
                self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            except ImportError:
                self.backbone = resnet50(pretrained=True)
        else:
            self.backbone = resnet50(weights=None)

        # Remove the original FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Freeze Backbone Logic
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ---------------------------------------------------------
        # 2. Architecture Improvements (Separate Heads + BatchNorm)
        # ---------------------------------------------------------
        
        # == Age Head (Regression) ==
        # زدنا طبقات عشان يقدر يفهم تعقيد العمر أكتر
        self.age_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),      # <--- بالإضافة الجديدة: استقرار للتدريب
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            
            nn.Linear(hidden_dim // 2, 1)    # Output: Age
        )

        # == Gender Head (Classification) ==
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),      # <--- بالإضافة الجديدة
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            
            nn.Linear(hidden_dim // 2, 2)    # Output: Gender Logits
        )

        # 3. Smart Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Smart Init for Age:
        # بنخلي أخر طبقة للعمر تبدأ بbias = 30، عشان الموديل يبدأ توقعه من 30 سنة مش من صفر.
        # ده بيسرع الـ Convergence جداً في الـ Regression.
        nn.init.constant_(self.age_head[-1].bias, 30.0)

    def forward(self, x):
        features = self.backbone(x)        # [Batch, 2048]
        
        # كل واحد بياخد مسار لوحده
        age = self.age_head(features)      # [Batch, 1]
        gender = self.gender_head(features)# [Batch, 2]
        
        return age, gender