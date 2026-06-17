import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

def imread_unicode(path):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)

class ConditionalLowLightDataset(Dataset):
    def __init__(self, low_dir, enh_dir, meta_file, transform=None, augment=False):
        self.low_dir = low_dir
        self.enh_dir = enh_dir
        self.transform = transform
        self.augment = augment
        self.aug = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        ])

        self.pairs = []

        # 하위폴더 존재 여부 확인
        subdirs = [d for d in os.listdir(enh_dir)
                   if os.path.isdir(os.path.join(enh_dir, d))]

        if subdirs:
            # 하위폴더 구조: enh_dir/sub/fname.png
            for sub in sorted(subdirs):
                enh_sub = os.path.join(enh_dir, sub)
                low_sub = os.path.join(low_dir, sub)
                if not os.path.isdir(low_sub):
                    continue
                for fname in sorted(os.listdir(enh_sub)):
                    if not fname.lower().endswith(('.jpg', '.png')) or fname.startswith('mask_'):
                        continue
                    lp = os.path.join(low_sub, fname)
                    ep = os.path.join(enh_sub, fname)
                    if os.path.exists(lp) and os.path.exists(ep):
                        self.pairs.append((lp, ep, sub, fname))
        else:
            # 평탄 구조: enh_dir/fname.png
            for fname in sorted(os.listdir(enh_dir)):
                if not fname.lower().endswith(('.jpg', '.png')) or fname.startswith('mask_'):
                    continue
                lp = os.path.join(low_dir, fname)
                ep = os.path.join(enh_dir, fname)
                if os.path.exists(lp) and os.path.exists(ep):
                    self.pairs.append((lp, ep, "", fname))

        print(f"[Dataset] 총 {len(self.pairs)}개 페어 로드됨")

        with open(meta_file) as f:
            self.meta = json.load(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, enh_path, sub, fname = self.pairs[idx]

        low = imread_unicode(low_path)
        enh_img = imread_unicode(enh_path)
        if low is None or enh_img is None:
            raise FileNotFoundError(f"로딩 실패: {low_path} 또는 {enh_path}")

        low_rgb = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
        enh_rgb = cv2.cvtColor(enh_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            low_t = self.transform(low_rgb)
            enh_t = self.transform(enh_rgb)
        else:
            low_t = torch.tensor(low_rgb).permute(2, 0, 1).float().div(255)
            enh_t = torch.tensor(enh_rgb).permute(2, 0, 1).float().div(255)

        if self.augment:
            low_t = self.aug(low_t)

        # metadata 키 탐색: fname → 확장자 제거 순으로 시도
        md = self.meta.get(fname) or self.meta.get(os.path.splitext(fname)[0])
        if md is None:
            brightness, color_shifts = 0.5, [0.0, 0.0, 0.0]
        else:
            brightness = md['brightness'] / 255.0
            color_shifts = [c / 255.0 for c in md['color_shift']]
        cond = torch.tensor([brightness] + color_shifts, dtype=torch.float32)

        # 마스크 로드 (없으면 전체 1)
        if sub:
            mask_path = os.path.join(self.enh_dir, sub, f"mask_{fname}")
        else:
            mask_path = os.path.join(self.enh_dir, f"mask_{fname}")

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.ones((256, 256), dtype=np.float32)
            else:
                mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(np.float32) / 255.0
        else:
            mask = np.ones((256, 256), dtype=np.float32)

        m_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return low_t, enh_t, cond, m_t