import os
import cv2
import numpy as np
import json

def analyze_and_generate_metadata(low_dir, enh_dir, save_name="metadata.json"):
    metadata = {}
    
    # 하위 폴더 탐색
    for sub in sorted(os.listdir(enh_dir)):
        enh_sub = os.path.join(enh_dir, sub)
        low_sub = os.path.join(low_dir, sub)
        
        if not os.path.isdir(enh_sub) or not os.path.isdir(low_sub):
            continue
            
        print(f"Processing subdirectory: {sub}")
        
        for fname in sorted(os.listdir(enh_sub)):
            if not fname.lower().endswith(('.jpg', '.png')) or fname.startswith('mask_'):
                continue
                
            low_path = os.path.join(low_sub, fname)
            enh_path = os.path.join(enh_sub, fname)
            
            if not os.path.exists(low_path):
                continue

            low_bgr = cv2.imread(low_path)
            enh_bgr = cv2.imread(enh_path)
            if low_bgr is None or enh_bgr is None:
                continue

            low_hsv = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            enh_hsv = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

            diff_rgb = cv2.absdiff(low_bgr, enh_bgr)
            mask = (cv2.cvtColor(diff_rgb, cv2.COLOR_BGR2GRAY) > 15).astype(np.uint8)*255

            V_low_px = low_hsv[...,2][mask>0]
            V_enh_px = enh_hsv[...,2][mask>0]
            if len(V_low_px) > 0:
                v_diff = float(np.mean(V_enh_px) - np.mean(V_low_px))
            else:
                v_diff = 0.0

            lo_hsv_adj = low_hsv.copy()
            lo_hsv_adj[...,2] = np.clip(lo_hsv_adj[...,2] + v_diff, 0, 255)
            lo_rgb_adj = cv2.cvtColor(lo_hsv_adj.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
            color_diff = np.mean(enh_bgr.astype(np.float32) - lo_rgb_adj, axis=(0,1)).tolist()

            metadata[fname] = {
                "brightness": v_diff,
                "color_shift": color_diff
            }
            # 마스크를 해당 서브폴더에 저장
            cv2.imwrite(os.path.join(enh_sub, f"mask_{fname}"), mask)

    with open(os.path.join(enh_dir, save_name), 'w') as f:
        json.dump(metadata, f, indent=4)
    print("✅ 메타데이터 및 마스크 생성 완료.")
