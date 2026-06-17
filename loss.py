import torch
import torch.nn as nn
import torchvision.models as models
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        try:
            vgg = models.vgg16(weights=weights).features[:9].eval()
        except RuntimeError:
            import os
            import torch.hub
            cache = torch.hub.get_dir()
            ckpt = weights.url.split('/')[-1]
            path = os.path.join(cache,'checkpoints',ckpt)
            if os.path.exists(path): os.remove(path)
            vgg = models.vgg16(weights=weights).features[:9].eval()
        for p in vgg.parameters(): p.requires_grad=False
        self.vgg = vgg
        self.crit = nn.MSELoss()

    def forward(self, x, y):
        return self.crit(self.vgg(x), self.vgg(y))
class ColorConstancyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        x: 모델이 출력한 복원 영상 [Batch, Channel, Height, Width]
        """
        mean_r = torch.mean(x[:, 0, :, :])
        mean_g = torch.mean(x[:, 1, :, :])
        mean_b = torch.mean(x[:, 2, :, :])
        
        # R-G, R-B, G-B 채널 평균 간의 차이의 제곱합
        loss = (mean_r - mean_g)**2 + (mean_r - mean_b)**2 + (mean_g - mean_b)**2
        return loss
