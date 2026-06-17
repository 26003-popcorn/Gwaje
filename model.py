import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 블록 (채널 어텐션)"""
    def __init__(self, ch, r=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch//r, 1), nn.ReLU(),
            nn.Conv2d(ch//r, ch, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.net(x)

class GlobalLocalFiLMBlock(nn.Module):
    """
    Global-Local Feature-wise Linear Modulation 블록
    - Global: 밝기/색상 시프트 조건(cond)을 채널별 스케일(Gamma)과 시프트(Beta)로 변환
    - Local: 에지/구조 맵(struct_map)을 공간적(Spatial) 스케일과 시프트로 변환
    """
    def __init__(self, ch, cond_dim=3):
        super().__init__()
        # Global FiLM 생성기 (조건 벡터 -> 채널 파라미터)
        self.global_fc = nn.Linear(cond_dim, ch * 2)
        
        # Local FiLM 생성기 (1채널 구조 맵 -> 공간 채널 파라미터)
        self.local_conv = nn.Conv2d(1, ch * 2, kernel_size=3, padding=1)
        
        # 초기 학습 안정화를 위해 가중치를 작게, 편향을 0으로 둡니다. (초기 출력 축소)
        nn.init.xavier_uniform_(self.global_fc.weight, gain=0.01)
        nn.init.zeros_(self.global_fc.bias)
        nn.init.xavier_uniform_(self.local_conv.weight, gain=0.01)
        nn.init.zeros_(self.local_conv.bias)

    def forward(self, x, cond, local_map):
        # 1. Global FiLM 파라미터 계산
        g_params = self.global_fc(cond).unsqueeze(-1).unsqueeze(-1) # [B, C*2, 1, 1]
        g_gamma, g_beta = torch.chunk(g_params, 2, dim=1)
        
        # 2. Local FiLM 파라미터 계산 (구조 맵 크기를 현재 특성 맵 크기에 맞춤)
        if local_map.shape[2:] != x.shape[2:]:
            local_map = F.interpolate(local_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        l_params = self.local_conv(local_map) # [B, C*2, H, W]
        l_gamma, l_beta = torch.chunk(l_params, 2, dim=1)
        
        # 3. Global과 Local의 융합 아핀 변환
        # 초기 상태에서는 gamma가 0에 가깝고 beta가 0에 가까워 x가 원본을 유지하며 안전하게 시작합니다.
        gamma = g_gamma + l_gamma
        beta = g_beta + l_beta
        return x * (1 + gamma) + beta

class UNetConditionalModel(nn.Module):
    def __init__(self, cond_dim=3):
        super().__init__()
        
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(),
                SEBlock(out_c)
            )
            
        # Encoder (조건 채널이 FiLM으로 주입되므로 입력은 순수 이미지 3채널)
        self.enc1 = block(3, 64)
        self.film1 = GlobalLocalFiLMBlock(64, cond_dim)
        
        self.enc2 = block(64, 128)
        self.film2 = GlobalLocalFiLMBlock(128, cond_dim)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bott = block(128, 256)
        self.film_bott = GlobalLocalFiLMBlock(256, cond_dim)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Decoder (스킵 커넥션 채널 결합 구조 반영)
        self.dec2 = block(256 + 128, 128)
        self.film_dec2 = GlobalLocalFiLMBlock(128, cond_dim)
        
        self.dec1 = block(128 + 64, 64)
        self.film_dec1 = GlobalLocalFiLMBlock(64, cond_dim)
        
        self.final = nn.Conv2d(64, 3, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.final.weight, gain=0.01)
        nn.init.zeros_(self.final.bias)

    def forward(self, x, cond, struct_map):
        # ── Encoder Stage ──
        e1 = self.enc1(x)
        e1 = self.film1(e1, cond, struct_map)
        
        e2 = self.enc2(self.pool(e1))
        e2 = self.film2(e2, cond, struct_map)
        
        # ── Bottleneck Stage ──
        bn = self.bott(self.pool(e2))
        bn = self.film_bott(bn, cond, struct_map)
        
        # ── Decoder Stage ──
        d2 = self.dec2(torch.cat([self.up(bn), e2], dim=1))
        d2 = self.film_dec2(d2, cond, struct_map)
        
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        d1 = self.film_dec1(d1, cond, struct_map)
        
        # Residual 범위를 -0.5 ~ 0.5로 제한
        return torch.tanh(self.final(d1)) * 0.5

class SimpleEdgeExtractor(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def laplacian(x):
    kernel = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32, device=x.device)
    kernel = kernel.view(1,1,3,3).repeat(x.size(1),1,1,1)
    return F.conv2d(x, kernel, padding=1, groups=x.size(1))
