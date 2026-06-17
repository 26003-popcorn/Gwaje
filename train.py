import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import kornia.color as KC
import kornia.losses as KL
import kornia.metrics as KM
import lpips

from dataset import ConditionalLowLightDataset
from model import UNetConditionalModel, SimpleEdgeExtractor, laplacian
from loss import VGGPerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 평가 지표를 위한 클래스 ---
class ImageEvaluator:
    def __init__(self, device):
        self.device = device
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(self.device).eval()
        for param in self.lpips_vgg.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def evaluate(self, pred, target):
        psnr_val = KM.psnr(pred, target, max_val=1.0).mean().item()
        ssim_val = KM.ssim(pred, target, window_size=11, max_val=1.0).mean().item()
        lpips_val = self.lpips_vgg((pred * 2.0) - 1.0, (target * 2.0) - 1.0).mean().item()
        return {"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val}

# --- 메인 학습 함수 ---
def train(low_dir, enh_dir, meta_file, epochs=1000, bs=10, lr=1e-4):
    x = [2.6667, 0.2414, 1.0, 1.5, 0.6316, 0.01108, 1.4286]
    
    transform = T.Compose([T.ToPILImage(), T.Resize((256,256)), T.ToTensor()])
    ds = ConditionalLowLightDataset(low_dir, enh_dir, meta_file, transform, augment=True)
    n_val = int(0.2 * len(ds))
    tr_ds, va_ds = random_split(ds, [len(ds) - n_val, n_val])
    tr = DataLoader(tr_ds, bs, shuffle=True)
    va = DataLoader(va_ds, bs)
    
    model = UNetConditionalModel().to(device)
    structure_model = SimpleEdgeExtractor().to(device)
    evaluator = ImageEvaluator(device)
    
    opt = optim.Adam(list(model.parameters()) + list(structure_model.parameters()), lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    perc = VGGPerceptualLoss().to(device)
    mse = nn.MSELoss()
    best = float('inf')
    pat = 0

    for e in range(epochs):
        model.train()
        structure_model.train()
        total_loss = 0
        
        for lo, eh, cond, msk in tr:
            lo, eh, cond, msk = lo.to(device), eh.to(device), cond.to(device), msk.to(device)
            b = cond[:, :1]
            cs = cond[:, 1:]

            lo_hsv = KC.rgb_to_hsv(lo)
            lo_hsv[:,2:3,:,:] = torch.clamp(lo_hsv[:,2:3,:,:] + b.view(-1,1,1,1) * msk, 0.0, 1.0)
            lo_b = KC.hsv_to_rgb(lo_hsv)
            lo_bc = torch.clamp(lo_b + cs.view(-1,3,1,1) * msk, 0.0, 1.0)

            opt.zero_grad()
            struct_map = structure_model(lo_bc)
            
            # 🚨 오류의 원인이었던 autocast와 scaler를 깔끔하게 제거했습니다!
            residual = model(lo_bc, cs, struct_map)
            out = torch.clamp(lo_bc + residual, 0.0, 1.0)
            
            l_mse = mse(out, eh) * x[0]
            l_per = perc(out, eh) * x[1]
            l_ssim = KL.ssim_loss(out, eh, window_size=11) * x[2]
            l_hf  = F.l1_loss(laplacian(out), laplacian(eh)) * x[3]
            
            hsv_out, hsv_gt = KC.rgb_to_hsv(out), KC.rgb_to_hsv(eh)
            l_sat = F.l1_loss(hsv_out[:,1:2], hsv_gt[:,1:2]) * x[4]
            
            lab_out, lab_gt = KC.rgb_to_lab(out), KC.rgb_to_lab(eh)
            l_lab = F.l1_loss(lab_out[:,1:], lab_gt[:,1:]) * x[5]
            
            tv_loss = (torch.abs(out[:,:,1:,:] - out[:,:,:-1,:]).mean() + 
                       torch.abs(out[:,:,:,1:] - out[:,:,:,:-1]).mean()) * x[6]
            
            loss = l_mse + l_per + l_ssim + l_hf + l_sat + l_lab + tv_loss
            
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # --- 검증 루프 ---
        model.eval()
        structure_model.eval()
        val_loss = 0
        total_psnr = total_ssim = total_lpips = 0.0
        
        with torch.no_grad():
            for lo, eh, cond, msk in va:
                lo, eh, cond, msk = lo.to(device), eh.to(device), cond.to(device), msk.to(device)
                b, cs = cond[:, :1], cond[:, 1:]
                
                lo_hsv = KC.rgb_to_hsv(lo)
                lo_hsv[:,2:3,:,:] = torch.clamp(lo_hsv[:,2:3,:,:] + b.view(-1,1,1,1) * msk, 0.0, 1.0)
                lo_b = KC.hsv_to_rgb(lo_hsv)
                lo_bc = torch.clamp(lo_b + cs.view(-1,3,1,1) * msk, 0.0, 1.0)
                
                struct_map = structure_model(lo_bc)
                residual = model(lo_bc, cs, struct_map)
                out = torch.clamp(lo_bc + residual, 0.0, 1.0)
                
                metrics = evaluator.evaluate(out, eh)
                total_psnr += metrics["psnr"]
                total_ssim += metrics["ssim"]
                total_lpips += metrics["lpips"]
                
                v_loss = mse(out, eh)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(va)
        avg_psnr = total_psnr / len(va)
        avg_ssim = total_ssim / len(va)
        avg_lpips = total_lpips / len(va)

        print(f"Epoch {e+1}/{epochs} | Train Loss: {total_loss/len(tr):.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f" >> [Metrics] PSNR: {avg_psnr:.2f}dB | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")

        # --- 모델 저장 ---
        scheduler.step(avg_val_loss)
        if avg_val_loss < best:
            torch.save(model.state_dict(), "best_model.pth")
            torch.save(structure_model.state_dict(), "best_structure.pth")
            best = avg_val_loss
            pat = 0
            print(" >> Best models saved.")
        else:
            pat += 1
            
        if pat > 15:
            print("Early stopping triggered.")
            break

    # 최종 저장
    torch.save(model.state_dict(), "final_model.pth")
    torch.save(structure_model.state_dict(), "final_structure.pth")
    print("Training finished.")