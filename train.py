import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import kornia.color as KC
import kornia.metrics as KM
import lpips

from dataset import ConditionalLowLightDataset
from model import UNetConditionalModel, SimpleEdgeExtractor
from loss import VGGPerceptualLoss, ColorConstancyLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 평가 지표를 위한 클래스 ---
class ImageEvaluator:
    def __init__(self, device):
        self.device = device
        self.lpips_vgg = lpips.LPIPS(net="vgg").to(self.device).eval()

        for param in self.lpips_vgg.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def evaluate(self, pred, target):
        psnr_val = KM.psnr(pred, target, max_val=1.0).mean().item()
        ssim_val = KM.ssim(pred, target, window_size=11, max_val=1.0).mean().item()
        lpips_val = self.lpips_vgg(
            (pred * 2.0) - 1.0,
            (target * 2.0) - 1.0
        ).mean().item()

        return {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val
        }


# --- 메인 학습 함수 ---
def train(low_dir, enh_dir, meta_file, epochs=1000, bs=10, lr=1e-4):
    # L_total = λ1·L_MSE + λ2·L_Perceptual + λ3·L_LPIPS + λ4·L_col
    lambda_mse = 2.6667
    lambda_perceptual = 0.2414
    lambda_lpips = 0.5
    lambda_col = 0.1

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    ds = ConditionalLowLightDataset(
        low_dir,
        enh_dir,
        meta_file,
        transform,
        augment=True
    )

    n_val = int(0.2 * len(ds))
    n_train = len(ds) - n_val

    tr_ds, va_ds = random_split(ds, [n_train, n_val])

    tr = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va = DataLoader(va_ds, batch_size=bs, shuffle=False)

    model = UNetConditionalModel().to(device)
    structure_model = SimpleEdgeExtractor().to(device)
    evaluator = ImageEvaluator(device)

    opt = optim.Adam(
        list(model.parameters()) + list(structure_model.parameters()),
        lr=lr
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=5
    )

    mse = nn.MSELoss()
    perceptual_loss = VGGPerceptualLoss().to(device).eval()
    color_constancy_loss = ColorConstancyLoss().to(device)

    lpips_loss = lpips.LPIPS(net="vgg").to(device).eval()
    for param in lpips_loss.parameters():
        param.requires_grad = False

    best = float("inf")
    pat = 0

    for e in range(epochs):
        model.train()
        structure_model.train()

        total_loss = 0.0

        for lo, eh, cond, msk in tr:
            lo = lo.to(device)
            eh = eh.to(device)
            cond = cond.to(device)
            msk = msk.to(device)

            b = cond[:, :1]
            cs = cond[:, 1:]

            lo_hsv = KC.rgb_to_hsv(lo)

            lo_hsv[:, 2:3, :, :] = torch.clamp(
                lo_hsv[:, 2:3, :, :] + b.view(-1, 1, 1, 1) * msk,
                0.0,
                1.0
            )

            lo_b = KC.hsv_to_rgb(lo_hsv)

            lo_bc = torch.clamp(
                lo_b + cs.view(-1, 3, 1, 1) * msk,
                0.0,
                1.0
            )

            opt.zero_grad()

            struct_map = structure_model(lo_bc)
            residual = model(lo_bc, cs, struct_map)

            out = torch.clamp(lo_bc + residual, 0.0, 1.0)

            l_mse = mse(out, eh) * lambda_mse

            l_perceptual = perceptual_loss(out, eh) * lambda_perceptual

            l_lpips = lpips_loss(
                (out * 2.0) - 1.0,
                (eh * 2.0) - 1.0
            ).mean() * lambda_lpips

            l_col = color_constancy_loss(out) * lambda_col

            loss = l_mse + l_perceptual + l_lpips + l_col

            loss.backward()
            opt.step()

            total_loss += loss.item()

        # --- 검증 루프 ---
        model.eval()
        structure_model.eval()

        val_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0

        with torch.no_grad():
            for lo, eh, cond, msk in va:
                lo = lo.to(device)
                eh = eh.to(device)
                cond = cond.to(device)
                msk = msk.to(device)

                b = cond[:, :1]
                cs = cond[:, 1:]

                lo_hsv = KC.rgb_to_hsv(lo)

                lo_hsv[:, 2:3, :, :] = torch.clamp(
                    lo_hsv[:, 2:3, :, :] + b.view(-1, 1, 1, 1) * msk,
                    0.0,
                    1.0
                )

                lo_b = KC.hsv_to_rgb(lo_hsv)

                lo_bc = torch.clamp(
                    lo_b + cs.view(-1, 3, 1, 1) * msk,
                    0.0,
                    1.0
                )

                struct_map = structure_model(lo_bc)
                residual = model(lo_bc, cs, struct_map)

                out = torch.clamp(lo_bc + residual, 0.0, 1.0)

                metrics = evaluator.evaluate(out, eh)

                total_psnr += metrics["psnr"]
                total_ssim += metrics["ssim"]
                total_lpips += metrics["lpips"]

                v_l_mse = mse(out, eh) * lambda_mse

                v_l_perceptual = perceptual_loss(out, eh) * lambda_perceptual

                v_l_lpips = lpips_loss(
                    (out * 2.0) - 1.0,
                    (eh * 2.0) - 1.0
                ).mean() * lambda_lpips

                v_l_col = color_constancy_loss(out) * lambda_col

                v_loss = v_l_mse + v_l_perceptual + v_l_lpips + v_l_col

                val_loss += v_loss.item()

        avg_train_loss = total_loss / len(tr)
        avg_val_loss = val_loss / len(va)

        avg_psnr = total_psnr / len(va)
        avg_ssim = total_ssim / len(va)
        avg_lpips = total_lpips / len(va)

        print(
            f"Epoch {e + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        print(
            f" >> [Metrics] "
            f"PSNR: {avg_psnr:.2f}dB | "
            f"SSIM: {avg_ssim:.4f} | "
            f"LPIPS: {avg_lpips:.4f}"
        )

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

    torch.save(model.state_dict(), "final_model.pth")
    torch.save(structure_model.state_dict(), "final_structure.pth")

    print("Training finished.")

    # 최종 저장
    torch.save(model.state_dict(), "final_model.pth")
    torch.save(structure_model.state_dict(), "final_structure.pth")
    print("Training finished.")
