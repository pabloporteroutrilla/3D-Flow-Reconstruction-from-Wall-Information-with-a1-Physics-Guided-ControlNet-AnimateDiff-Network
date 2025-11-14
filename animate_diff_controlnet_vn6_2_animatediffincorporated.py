#####################################################################################################
#####################################################################################################
#####                                       CONTROLNET + ANIMATEDIFF                            #####
#####################################################################################################
#####################################################################################################
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

#from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path

import numpy as np
import h5py    
import math
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib import rc, rcParams
import matplotlib.colors as colors
from matplotlib import ticker
import matplotlib.cm as cm
import glob, re, math

rc('text', usetex=True)
rc('font', family='serif')

#############################################################
#####              i) Dataset Definition                #####
#############################################################
class ChannelFlowDataset(Dataset):
    def __init__(self, file_paths, minsmaxs,
                 x_key="X_features", y_key="Y_features",
                 dtype=np.float32, read_half=False,  # read_half=True intentará np.float16
                 keep_device=None):                  # mejor None: mover a GPU en el loop
        self.file_paths = list(file_paths)
        self.x_key = x_key
        self.y_key = y_key
        self.mm = minsmaxs
        self.dtype = np.float16 if read_half else dtype
        self.keep_device = keep_device

        # sanity de escalas
        for k in ("P_min","P_max","tau_w_x_min","tau_w_x_max","tau_w_z_min","tau_w_z_max","u_min","u_max","v_min","v_max","w_min","w_max"):
            assert k in self.mm, f"Falta {k} en minsmaxs"

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def _norm(arr, lo, hi):
        return (arr - lo) / ( (hi - lo) + 1e-12 )

    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        # rdcc_*: caché de chunks para acelerar I/O; swmr=True para lecturas concurrentes
        with h5py.File(fp, "r", swmr=True,
                       rdcc_nbytes=256*1024**2, rdcc_nslots=1_000_003) as f:
            X = f[self.x_key][...]  # (3, H+2, W+2)
            Y = f[self.y_key][...]  # (C, H+2, Ny, W+2)

        #  Elimination of the ghost-cells
        X = X[:, 1:-1, 1:-1]          # (3,H,W)
        Y = Y[:, 1:-1, :, 1:-1]       # (3,H,Ny,W)

        # Reduce RAM/PCIe si read_half=True
        X = X.astype(self.dtype, copy=False)
        Y = Y.astype(self.dtype, copy=False)

        # Normalize X_features
        X[0] = self._norm(X[0], self.mm["P_min"],  self.mm["P_max"])
        X[1] = self._norm(X[1], self.mm["tau_w_x_min"], self.mm["tau_w_x_max"])
        X[2] = self._norm(X[2], self.mm["tau_w_z_min"], self.mm["tau_w_z_max"])

        # Normalize Y_features
        Y[0] = self._norm(Y[0], self.mm["u_min"], self.mm["u_max"])
        Y[1] = self._norm(Y[1], self.mm["v_min"], self.mm["v_max"])
        Y[2] = self._norm(Y[2], self.mm["w_min"], self.mm["w_max"])


        X_t = torch.from_numpy(X)
        Y_t = torch.from_numpy(Y)

        if self.keep_device is not None:
            X_t = X_t.to(self.keep_device, non_blocking=True)
            Y_t = Y_t.to(self.keep_device, non_blocking=True)

        return X_t, Y_t


#############################################################
#####                Physics Informed                   #####
#############################################################

# Physics Implemented in the Network: 
# - Navier-Stokes Equation: Momentum
# - Navier-Stokes Equation: Continuity 
# - Periodic Boundary Condition


######################################################################
#####      Stable Diffusion U-Net Encoder & Decoder Blocks      ######
######################################################################
class SDEncoderBlockA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.act1  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act2  = nn.SiLU()
    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        return self.act2(self.norm2(self.conv2(x)))

class SDEncoderBlockB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_ds = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm_ds = nn.GroupNorm(32, out_ch)
        self.act_ds  = nn.SiLU()
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2   = nn.GroupNorm(32, out_ch)
        self.act2    = nn.SiLU()
    def forward(self, x):
        x = self.act_ds(self.norm_ds(self.conv_ds(x)))
        return self.act2(self.norm2(self.conv2(x)))

class SDEncoderBlockC(SDEncoderBlockB):
    pass  # downsample 32->16

class SDEncoderBlockD(SDEncoderBlockB):
    pass  # downsample 16->8

class SDMiddleBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, ch)
        self.act1  = nn.SiLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, ch)
        self.act2  = nn.SiLU()
    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        return self.act2(self.norm2(self.conv2(x)))

class SDDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.act1  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act2  = nn.SiLU()
    def forward(self, x, skip):
        x = self.act1(self.norm1(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        return self.act2(self.norm2(self.conv2(x)))
    
class SDDecoderFinal(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act  = nn.SiLU()
    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        return self.act(self.norm(self.conv(x)))


class ZeroConv1x1(nn.Conv2d):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, 1)
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

##################################################################
#####                 Discriminator 2D                      ######
################################################################## 
class PatchDiscriminator2D(nn.Module):
    def __init__(self, in_ch=9, base=64):
        super().__init__()
        def blk(ci, co, stride):
            return nn.Sequential(
                nn.Conv2d(ci, co, 4, stride=stride, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        # Sin BN en el primero; usa SpectralNorm si quieres aún más estabilidad
        self.net = nn.Sequential(
            blk(in_ch,   base,   2),              # H/2
            nn.Conv2d(base, base*2, 4, 2, 1),     # H/4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 4, 2, 1),   # H/8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*4, base*8, 4, 1, 1),   # H/8 (más profundo)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*8, 1, 4, 1, 1)         # mapa de patches
        )

    def forward(self, x):
        # x: (B, 7, H, W)
        return self.net(x)  # (B,1,H',W') logits

def build_D_input(slice_uv, cond2d, prev_uv):
    # todos (B,*,H,W) → concat por canales
    return torch.cat([slice_uv, cond2d, prev_uv], dim=1)  # (B,7,H,W)

def d_hinge_loss(real_logits, fake_logits):
    # E[relu(1 - real)] + E[relu(1 + fake)]
    return (F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean())

def g_hinge_loss(fake_logits):
    # -E[fake]
    return (-fake_logits.mean())

###########################################################################################
######       ControlNetFull: complete encoder+decoder with zero-conv injections       #####
###########################################################################################
class ControlNetFull(nn.Module):
    def __init__(self, in_ch=3, cond_ch=3, base_ch=48):
        super().__init__()
        self.base_ch = base_ch
        # Encoder/decoder base (U-Net)
        self.base_A   = SDEncoderBlockA(in_ch, base_ch)          # (B,3,H,W)->(B,32,H,W)
        self.base_B   = SDEncoderBlockB(base_ch, 2*base_ch)      # H/2
        self.base_C   = SDEncoderBlockC(2*base_ch, 4*base_ch)    # H/4
        self.base_D   = SDEncoderBlockD(4*base_ch, 8*base_ch)    # H/8
        self.base_mid = SDMiddleBlock(8*base_ch)
        self.base_dD  = SDDecoderBlock(8*base_ch, 4*base_ch, 4*base_ch)
        self.base_dC  = SDDecoderBlock(4*base_ch, 2*base_ch, 2*base_ch)
        self.base_dB  = SDDecoderBlock(2*base_ch,     base_ch,       base_ch)
        self.base_dA  = SDDecoderFinal(base_ch,       base_ch,       3)  # salida (u,v,w)

        # Rama "control" (P, tauwx, tauwz)
        self.ctrl_A   = SDEncoderBlockA(cond_ch, base_ch)
        self.ctrl_B   = SDEncoderBlockB(base_ch, 2*base_ch)
        self.ctrl_C   = SDEncoderBlockC(2*base_ch, 4*base_ch)
        self.ctrl_D   = SDEncoderBlockD(4*base_ch, 8*base_ch)
        self.ctrl_mid = SDMiddleBlock(8*base_ch)

        # Inyecciones ZeroConv (inicializadas a cero)
        self.zc_A   = ZeroConv1x1(base_ch,     base_ch)
        self.zc_B   = ZeroConv1x1(2*base_ch,   2*base_ch)
        self.zc_C   = ZeroConv1x1(4*base_ch,   4*base_ch)
        self.zc_D   = ZeroConv1x1(8*base_ch,   8*base_ch)
        self.zc_mid = ZeroConv1x1(8*base_ch,   8*base_ch)
        self.zc_dD  = ZeroConv1x1(4*base_ch,   4*base_ch)
        self.zc_dC  = ZeroConv1x1(2*base_ch,   2*base_ch)
        self.zc_dB  = ZeroConv1x1(base_ch,     base_ch)
        self.zc_dA  = ZeroConv1x1(base_ch,     3)

        self.motion = MotionTransformer(ch=8*base_ch, num_layers=4, num_heads=8, ff_mult=4)


    def _encode_prev(self, prev_uv):
        xA = self.base_A(prev_uv)
        xB = self.base_B(xA)
        xC = self.base_C(xB)
        xD = self.base_D(xC)
        xm = self.base_mid(xD)
        return xA, xB, xC, xm

    def _encode_cond(self, cond2d):
        cA = self.ctrl_A(cond2d)
        cB = self.ctrl_B(cA)
        cC = self.ctrl_C(cB)
        cD = self.ctrl_D(cC)
        cm = self.ctrl_mid(cD)
        return cA, cB, cC, cm

    # --- Inyección tipo ControlNet ---
    def _inject(self, xA, xB, xC, xm, cA, cB, cC, cm, guidance_scale):
        xA = xA + self.zc_A(cA)   * guidance_scale
        xB = xB + self.zc_B(cB)   * guidance_scale
        xC = xC + self.zc_C(cC)   * guidance_scale
        xm = xm + self.zc_mid(cm) * guidance_scale
        return xA, xB, xC, xm

    # --- Decoder (con inyección en el decodificador) ---
    def _decode(self, xA, xB, xC, xm, cA, cB, cC, guidance_scale):
        dD = self.base_dD(xm, xC); dD = dD + self.zc_dD(cC) * guidance_scale
        dC = self.base_dC(dD, xB); dC = dC + self.zc_dC(cB) * guidance_scale
        dB = self.base_dB(dC, xA); dB = dB + self.zc_dB(cA) * guidance_scale
        out = self.base_dA(dB, xA); out = out + self.zc_dA(cA) * guidance_scale
        return out
    
    def forward(self, prev_uv, cond2d, guidance_scale: float = 1.0):
        # single-step (B,2,H,W) -> (B,2,H,W)
        xA, xB, xC, xm = self._encode_prev(prev_uv)
        cA, cB, cC, cm = self._encode_cond(cond2d)
        xA, xB, xC, xm = self._inject(xA, xB, xC, xm, cA, cB, cC, cm, guidance_scale)
        out = self._decode(xA, xB, xC, xm, cA, cB, cC, guidance_scale)
        return out

        

    def forward_seq(self, prev_seq, cond2d, guidance_scale: float = 1.0, use_motion: bool = True):
        """
        prev_seq : (B, L, 3, H, W)   slices y⁺=i ... y⁺=i+L-1
        cond2d   : (B, 3, H, W)
        return   : (B, L, 3, H, W)   pred para y⁺=i+1 ... y⁺=i+L
        """
        B, L, _, H, W = prev_seq.shape

        # cond: se calcula una vez
        cA, cB, cC, cm = self._encode_cond(cond2d)

        # encode de cada slice previo
        xA_list, xB_list, xC_list, xm_list = [], [], [], []
        for i in range(L):
            xA_i, xB_i, xC_i, xm_i = self._encode_prev(prev_seq[:, i])
            xA_list.append(xA_i); xB_list.append(xB_i); xC_list.append(xC_i); xm_list.append(xm_i)

        # (B, L, Cm, H', W')
        xm_seq = torch.stack(xm_list, dim=1)

        # AnimateDiff en el "mid" (secuencia en la dimensión L)
        if use_motion:
            delta = self.motion(xm_seq)  # (B, L, Cm, H', W')
            xm_seq = xm_seq + delta

        # decode slice-a-slice
        outs = []
        for i in range(L):
            xA_i, xB_i, xC_i = xA_list[i], xB_list[i], xC_list[i]
            xm_i = xm_seq[:, i]
            xA_i, xB_i, xC_i, xm_i = self._inject(xA_i, xB_i, xC_i, xm_i, cA, cB, cC, cm, guidance_scale)
            out_i = self._decode(xA_i, xB_i, xC_i, xm_i, cA, cB, cC, guidance_scale)
            outs.append(out_i)

        return torch.stack(outs, dim=1)  # (B, L, 3, H, W)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int):
    """
    timesteps: (B,) enteros [0…T]
    dim: dimensión del embedding (debe ser par)
    devuelve: (B, dim)
    """
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
    # (B,1) * (half,) -> (B, half)
    emb = timesteps[:, None].float() * emb[None, :]
    # intercalar sin/cos
    emb = torch.cat([emb.sin(), emb.cos()], dim=1)  # (B, dim)
    return emb

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int, mlp_hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, emb_dim),
        )

    def forward(self, t: torch.Tensor):
        # t: (B,) de índices de slice
        emb = sinusoidal_embedding(t, self.mlp[0].in_features)
        return self.mlp(emb)  
    

##################################################################
#####         AnimateDiff  Motion Transformer               ######
################################################################## 
class MotionTransformer(nn.Module):
    def __init__(self, ch, num_layers=4, num_heads=8, ff_mult=4):
        super().__init__()
        # batch_first=True nos simplifica las formas (B*HW, N, C)
        enc = nn.TransformerEncoderLayer(d_model=ch, nhead=num_heads, dim_feedforward=ch*ff_mult, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.out_lin = nn.Linear(ch, ch)
        nn.init.zeros_(self.out_lin.weight)
        nn.init.zeros_(self.out_lin.bias)

    def forward(self, z_mid):  # z_mid: (B, N, C, H, W)
        B, N, C, H, W = z_mid.shape
        x = z_mid.permute(0, 3, 4, 1, 2).reshape(B*H*W, N, C)  # (BHW, N, C)

        # Positional encoding sinusoidal on-the-fly (N puede cambiar)
        pe = sinusoidal_embedding(torch.arange(N, device=z_mid.device), C).unsqueeze(0)  # (1,N,C)
        x = x + pe

        x = self.transformer(x)            # (BHW, N, C)
        delta = self.out_lin(x)            # (BHW, N, C)

        delta = delta.reshape(B, H, W, N, C).permute(0, 3, 4, 1, 2)  # (B,N,C,H,W)
        return delta


###################################################################################
######                      Training Loop Definition                          #####
################################################################################### 
# Only ControlNet
'''
def train_one_epoch(model, D, loader, optG, optD, schedulerG, device, guidance_scale, lambda_adv=1e-3):
    model.train(); D.train()
    data_meter, adv_meter, d_meter = 0.0, 0.0, 0.0

    for X, Y in loader:
        X, Y = X.to(device, non_blocking=True), Y.to(device,non_blocking=True)               # X:(B,3,H,W)  Y:(B,2,H,Ny,W)
        B, C, H, Ny, W = Y.shape

        for i in range(Ny-1):
            prev   = Y[:, :, :, i, :]      # (B,2,H,W) teacher forcing
            target = Y[:, :, :, i+1, :]    # (B,2,H,W)
            # ---------- G forward ----------
            pred = model(prev, X, guidance_scale)   # (B,2,H,W)

            # ---------- Update D ----------
            with torch.no_grad():
                fake_detached = pred.clamp(0, 1) if pred.dtype.is_floating_point else pred
            D_real_in = torch.cat([target, X, prev], dim=1)          # (B,7,H,W)
            D_fake_in = torch.cat([fake_detached, X, prev], dim=1)   # (B,7,H,W)
            real_logits = D(D_real_in)
            fake_logits = D(D_fake_in)
            d_loss = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()

            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()

            # ---------- Update G ----------
            l_data = F.l1_loss(pred, target)              # o MSE
            fake_logits_G = D(torch.cat([pred, X, prev], dim=1))
            g_adv = -fake_logits_G.mean()
            g_loss = l_data + lambda_adv * g_adv

            optG.zero_grad(set_to_none=True)
            g_loss.backward()
            optG.step()

            # meters
            data_meter += l_data.item()
            adv_meter += (lambda_adv * g_adv).item()
            d_meter   += d_loss.item()
        
        if schedulerG is not None: 
            schedulerG.step()

    # promedios por slice
    steps = len(loader) * (Ny-1)
    return {
        "data": data_meter / steps,
        "adv": adv_meter / steps,
        "d":   d_meter   / steps
    }


@torch.no_grad()
def validate(model, loader, device, guidance_scale, free_run=False):
    model.eval()
    total_loss, total_count = 0.0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        B, C, H, Ny, W = Y.shape
        prev = Y[:, :, :, 0, :]   # o una BC
        for i in range(Ny-1):
            pred   = model(prev, X, guidance_scale)
            target = Y[:, :, :, i+1, :]
            total_loss  += F.mse_loss(pred, target, reduction='sum').item()
            total_count += target.numel()
            prev = pred if free_run else Y[:, :, :, i, :]
    return total_loss / total_count


def fit(model, D, train_loader, val_loader, optG, optD, schedulerG,
        device, guidance_scale, num_epochs, output_dir, lambda_adv=1e-3):
    best_val = float('inf'); os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, num_epochs+1):
        tr = train_one_epoch(model, D, train_loader, optG, optD, schedulerG, device, guidance_scale, lambda_adv)
        val = validate(model, val_loader, device, guidance_scale, free_run=False)

        print(f"[{epoch}/{num_epochs}] data:{tr['data']:.6f} adv:{tr['adv']:.6f} D:{tr['d']:.6f} | ValMSE:{val:.6e}")

        if val < best_val:
            best_val = val
            ckpt = {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "D_state":         D.state_dict(),
                "optG_state":      optG.state_dict(),
                "optD_state":      optD.state_dict(),
                "best_val_mse":    best_val,
            }
            path = os.path.join(output_dir, f"best_epoch{epoch}.pt")
            torch.save(ckpt, path)
            print(f" ↪ Guardado: jofre@metafluid.eebe.upc.edu:{path}")
        torch.cuda.empty_cache()
    print("Entrenamiento finalizado. Mejor Val MSE =", best_val)
'''

# ControlNet+AnimateDiff
def train_one_epoch_seq(model, D, loader, optG, optD, schedulerG,
                        device, guidance_scale, lambda_adv=1e-3,
                        L=8, stride=8):
    """
    Entrena con ventanas secuenciales de longitud L (AnimateDiff ON).
    - X: (B,3,H,W)
    - Y: (B,2,H,Ny,W)
    Creamos Y_seq: (B,Ny,2,H,W), y generamos ventanas
      prev_seq   = Y_seq[:, s:s+L]            -> (B,L,2,H,W)
      target_seq = Y_seq[:, s+1:s+1+L]        -> (B,L,2,H,W)
    """
    model.train(); D.train()
    data_meter, adv_meter, d_meter = 0.0, 0.0, 0.0

    for X, Y in loader:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)             # (B,3,H,Ny,W)
        B, _, H, Ny, W = Y.shape
        Y_seq = Y.permute(0, 3, 1, 2, 4).contiguous()   # (B,Ny,2,H,W)

        Y_all  = Y                                    # (B,3,H,Ny,W)  target de 3C
        Yseq_all = Y_all.permute(0, 3, 1, 2, 4).contiguous()  # (B,Ny,3,H,W)

        # barrido por ventanas
        for s in range(0, Ny-1, stride):
            e = min(s + L, Ny-1)        
            L_eff = e - s               

            prev_seq   = Y_seq[:, s:e]          # (B,L_eff,3,H,W)
            target_seq = Y_seq[:, s+1:e+1]      # (B,L_eff,3,H,W)

            # ---------- G forward (AnimateDiff ON) ----------
            pred_seq = model.forward_seq(prev_seq, X, guidance_scale, use_motion=True)  # (B,L_eff,3,H,W)

            # ---------- UPDATE D (por-slice y acumulado) ----------
            d_loss_acc = 0.0
            with torch.no_grad():
                pred_seq_det = pred_seq.detach()

            for i in range(L_eff):
                prev_i   = prev_seq[:, i]       # (B,3,H,W)       
                targ_i   = target_seq[:, i]     # (B,3,H,W)  
                fake_i   = pred_seq_det[:, i]   # (B,3,H,W)  
                D_real_in = torch.cat([targ_i, X, prev_i], dim=1)  # (B,8,H,W)
                D_fake_in = torch.cat([fake_i, X, prev_i], dim=1)  # (B,8,H,W)
                real_logits = D(D_real_in)
                fake_logits = D(D_fake_in)
                d_loss_acc += (F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean())

            d_loss = d_loss_acc / L_eff
            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()

            # ---------- UPDATE G (data + adv por-slice) ----------
            l_data = F.l1_loss(pred_seq, target_seq)    # o MSE

            fake_logits_list = []
            for i in range(L_eff):
                prev_i = prev_seq[:, i]
                fake_i = pred_seq[:, i]
                fake_logits_G = D(torch.cat([fake_i, X, prev_i], dim=1))
                fake_logits_list.append(fake_logits_G)
            
            g_adv = -torch.stack(fake_logits_list).mean()
            g_loss = l_data + lambda_adv * g_adv

            optG.zero_grad(set_to_none=True)
            g_loss.backward()
            optG.step()

            data_meter += l_data.item()
            adv_meter  += (lambda_adv * g_adv).item()
            d_meter    += d_loss.item()

        if schedulerG is not None:
            schedulerG.step()

    # promedios por batch (si quieres por-slice, divide por nº total de ventanas)
    n_batches = len(loader)
    return {"data": data_meter / n_batches,
            "adv":  adv_meter  / n_batches,
            "d":    d_meter    / n_batches}

@torch.no_grad()
def validate_seq(model, loader, device, guidance_scale,
                 L=8, stride=8, free_run=False):
    """
    Eval secuencial. Si free_run=True, encadenas predicciones;
    si False, teacher forcing dentro de la ventana.
    """
    model.eval()
    total_loss, total_count = 0.0, 0

    for X, Y in loader:
        X = X.to(device); Y = Y.to(device)
        B, _, H, Ny, W = Y.shape
        Y_seq = Y.permute(0, 3, 1, 2, 4).contiguous()  # (B,Ny,3,H,W)

        for s in range(0, Ny-1, stride):
            e = min(s + L, Ny-1)
            L_eff = e - s

            prev_seq   = Y_seq[:, s:e]       # (B,L_eff,3,H,W)
            target_seq = Y_seq[:, s+1:e+1]

            if free_run:
                # arranca con la primera capa real y free-run dentro de la ventana
                preds = []
                prev = prev_seq[:, 0:1]                                         # (B,1,2,H,W)
                pred = model.forward_seq(prev, X, guidance_scale, True)[:, 0]   # (B,3,H,W)
                preds.append(pred)
                for i in range(1, L_eff):
                    prev = pred.unsqueeze(1)                 # usa tu propia pred como entrada
                    pred = model.forward_seq(prev, X, guidance_scale, True)[:, 0]
                    preds.append(pred)
                pred_seq = torch.stack(preds, dim=1)         # (B,L_eff,3,H,W)
            else:
                pred_seq = model.forward_seq(prev_seq, X, guidance_scale, True)

            total_loss  += F.mse_loss(pred_seq, target_seq, reduction='sum').item()
            total_count += target_seq.numel()

    return total_loss / total_count

def fit(model, D, train_loader, val_loader, optG, optD, schedulerG,
        device, guidance_scale, num_epochs, output_dir,
        lambda_adv=1e-3, L=8, stride=8):
    best_val = float('inf'); os.makedirs(output_dir, exist_ok=True)
    for epoch in range(1, num_epochs+1):
        tr = train_one_epoch_seq(model, D, train_loader, optG, optD, schedulerG,
                                 device, guidance_scale, lambda_adv, L=L, stride=stride)
        val = validate_seq(model, val_loader, device, guidance_scale,
                           L=L, stride=stride, free_run=False)

        print(f"[{epoch}/{num_epochs}] data:{tr['data']:.6f} adv:{tr['adv']:.6f} D:{tr['d']:.6f} | ValMSE:{val:.6e}")

        if val < best_val:
            best_val = val
            ckpt = {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "D_state":         D.state_dict(),
                "optG_state":      optG.state_dict(),
                "optD_state":      optD.state_dict(),
                "best_val_mse":    best_val,
            }
            path = os.path.join(output_dir, f"best_epoch_{epoch}_test_6.pt")
            torch.save(ckpt, path)
            print(f" ↪ Guardado: {path}")
        torch.cuda.empty_cache()
    print("Entrenamiento finalizado. Mejor Val MSE =", best_val)

def load_pretrained_partial(model, checkpoint_path, device):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)

    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if (k in model_state and v.shape == model_state[k].shape)}
    skipped = [k for k, v in state.items()
               if not (k in model_state and v.shape == model_state[k].shape)]
    missing = [k for k in model_state.keys() if k not in filtered]

    model.load_state_dict(filtered, strict=False)
    print(f"[pretrained] loaded {len(filtered)} tensors; "
          f"skipped {len(skipped)} (shape mismatch, e.g. first conv & head); "
          f"new/missing {len(missing)}.")



def build_model(device, use_pretrained: bool = False, checkpoint_path: str = None):
    model = ControlNetFull(in_ch=3, cond_ch=3, base_ch=32).to(device)
    if use_pretrained and checkpoint_path is not None:
        load_pretrained_partial(model, checkpoint_path, device)
    else:
        print("Inicializando la red desde cero.")
    return model



if __name__ == "__main__":
    #########################################################
    ###           i) PATHS & HYPERPARAMETERS              ###   
    #########################################################
    
    # A) Paths
    samples_dir = Path("/home/jofre/Members/Pablo/Paper1_Pablo/Samples")
    checkpoints_dir  = Path("/home/jofre/Members/Pablo/Paper1_Pablo/checkpoints")
    checkpoints_path = checkpoints_dir / "best_epoch_1_test_5.pt"
    mesh_dir = Path("/home/jofre/Members/Pablo/Paper1_Pablo/Snapshots")
    mesh_path = mesh_dir / "3d_turbulent_channel_flow-MESH.h5"

    #samples_dir      = Path("/home/pabloportero/Members/Pablo/Samples")
    #checkpoints_dir  = Path("/home/pabloportero/Members/Pablo/Paper1_Pablo/Code/checkpoints")
    #checkpoints_path = checkpoints_dir / "best_epoch640.pt"

    # B) Paramaters
    use_pretrained = True           # Use pre-trained weights
    guidance_scale = 3.0            # Guidance Scale AnimateDiff
    num_epochs = 650                # Number of Epochs
    batch_size = 8                  # Batch Size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = 8
    stride = 8


    # C) Setting of the Case
    #########################################################
    ###       3D INCOMPRESIBLE CHANNEL FLOW SETTUP        ###   
    #########################################################
    R_specific = 287.058                                        # Specific gas constant
    gamma_0    = 1.4                                            # Heat capacity ratio
    c_p        = gamma_0*R_specific/( gamma_0 - 1.0 )           # Isobaric heat capacity
    delta      = 1.0                                            # Channel half-height
    Re_tau     = 180.0                                          # Friction Reynolds number
    Ma         = 3.0e-1                                         # Mach number
    Pr         = 0.71                                           # Prandtl number
    rho_0      = 1.0                                            # Reference density
    u_tau      = 1.0                                            # Friction velocity
    tau_w      = rho_0*u_tau*u_tau                              # Wall shear stress
    mu         = rho_0*u_tau*delta/Re_tau                       # Dynamic viscosity 
    nu         = u_tau*delta/Re_tau                             # Kinematic viscosity 
    kappa      = c_p*mu/Pr                                      # Thermal conductivity        
    Re_b       = pow( Re_tau/0.09, 1.0/0.88 )                   # Bulk (approximated) Reynolds number
    u_b        = nu*Re_b/( 2.0*delta )                          # Bulk (approximated) velocity
    P_0        = rho_0*u_b*u_b/( gamma_0*Ma*Ma )                # Reference pressure
    T_0        = P_0/( rho_0*R_specific )                       # Reference temperature
    L_x        = 4.0*np.pi*delta                                # Streamwise length
    L_y        = 2.0*delta                                      # Wall-normal height
    L_z        = 4.0*np.pi*delta/3.0                            # Spanwise width
    kappa_vK   = 0.41;                                          # von Kármán constant
                                   
    
    #########################################################
    ###         ii) DATASET & DATALOADER SETTUP           ###   
    #########################################################
    #file_indices = list(range(280000, 5720001, 10000)) # 280000 TO 5720001
    file_indices  = list(range(280000, 43150001, 10000)) # 280000 TO 354100011
    print('Num of samples for the training:',len(range(1280000, 43150001, 10000)))    

    data_files0 = [f"datapost0_3d_turbulent_channel_flow_{i}.h5" for i in file_indices]
    X_features_list = []
    Y_features_list = []
    P_min = 0; P_max = 0
    tau_w_x_min = 0; tau_w_x_max = 0
    tau_w_z_min = 0; tau_w_z_max = 0
    u_min = 0; u_max = 0
    v_min = 0; v_max = 0 
    w_min = 0; w_max = 0
    for file in data_files0:
        file_path = os.path.join(samples_dir, file)
        with h5py.File(file_path, 'r') as data_file:
            X_data0_i = data_file['X_features'][:,:,:]
            Y_data0_i = data_file['Y_features'][:,:,:]
            X_data0_i = np.array(X_data0_i)
            Y_data0_i = np.array(Y_data0_i)
            # Computation of MAX & MIN values
            # P_min Computation
            if X_data0_i[0,1:-1,1:-1].min() < P_min or P_min ==0:
                P_min = X_data0_i[0,1:-1,1:-1].min()
            # P_max Computation
            if X_data0_i[0,1:-1,1:-1].max() > P_max or P_max ==0:
                P_max = X_data0_i[0,1:-1,1:-1].max()
            # tau_w_x_min Computation
            if X_data0_i[1,1:-1,1:-1].min() < tau_w_x_min or tau_w_x_min ==0:
                tau_w_x_min = X_data0_i[1,1:-1,1:-1].min()
            # tau_w_x_max Computation
            if X_data0_i[1,1:-1,1:-1].max() > tau_w_x_max or tau_w_x_max ==0:
                tau_w_x_max = X_data0_i[1,1:-1,1:-1].max()
            # tau_w_z_min Computation
            if X_data0_i[2,1:-1,1:-1].min() < tau_w_z_min or tau_w_z_min ==0:
                tau_w_z_min = X_data0_i[2,1:-1,1:-1].min()
            # tau_w_z_max Computation
            if X_data0_i[2,1:-1,1:-1].max() > tau_w_z_max or tau_w_z_max ==0:
                tau_w_z_max = X_data0_i[2,1:-1,1:-1].max()

            # u_min Computation
            if Y_data0_i[0,1:-1,:,1:-1].min() < u_min or u_min ==0:
                u_min = Y_data0_i[0,1:-1,1:-1].min()
            # u_max Computation
            if Y_data0_i[0,1:-1,:,1:-1].max() > u_max or u_max ==0:
                u_max = Y_data0_i[0,1:-1,1:-1].max()
                # v_min Computation
            if Y_data0_i[1,1:-1,:,1:-1].min() < v_min or v_min ==0:
                v_min = Y_data0_i[1,1:-1,1:-1].min()
            # v_max Computation
            if Y_data0_i[1,1:-1,:,1:-1].max() > v_max or v_max ==0:
                v_max = Y_data0_i[1,1:-1,1:-1].max()
            # w_min Computation
            if Y_data0_i[2,1:-1,:,1:-1].min() < w_min or w_min ==0:
                w_min = Y_data0_i[2,1:-1,1:-1].min()
            # w_max Computation
            if Y_data0_i[2,1:-1,:,1:-1].max() > w_max or w_max ==0:
                w_max = Y_data0_i[2,1:-1,1:-1].max()

    # MAX & MIN VALUES of P_w, tau_wx, tau_wz, u,v & w
    print('MAX & MIN VALUES of P_w, tau_wx, tau_wz, u,v & w')
    print('P_min',P_min)
    print('P_max',P_max)
    print('tau_w_x_min',tau_w_x_min)
    print('tau_w_x_max',tau_w_x_max)
    print('tau_w_z_min',tau_w_z_min)
    print('tau_w_z_max',tau_w_z_max)
    print('u_min',u_min)
    print('u_max',u_max)
    print('v_min',v_min)
    print('v_max',v_max)
    print('w_min',w_min)
    print('w_max',w_max)

    minsmaxs = dict(P_min=P_min,P_max=P_max,tau_w_x_min=tau_w_x_min,tau_w_x_max=tau_w_x_max,tau_w_z_min=tau_w_z_min,tau_w_z_max=tau_w_z_max,
                    u_min=u_min,u_max=u_max,v_min=v_min,v_max=v_max,w_min=w_min,w_max=w_max)
    
    file_paths = [str(samples_dir / f) for f in data_files0]

    ####################################################
    ###            Physics Implementation            ###
    ####################################################
    # Load Mesh 
    with h5py.File(mesh_path, 'r') as f:
            x_data = f['x'][:]  
            y_data = f['y'][:]
            z_data = f['z'][:]
    x_data = torch.from_numpy(x_data).to(device)  # [Z, Y, X]
    y_data = torch.from_numpy(y_data).to(device)
    z_data = torch.from_numpy(z_data).to(device)

    num_points_x = len(x_data[0,0,:])
    num_points_y = len(y_data[0,:,0])
    num_points_z = len(z_data[:,0,0])

    
    # Dataset & Dataloader
    dataset = ChannelFlowDataset(
        file_paths=file_paths,
        minsmaxs=minsmaxs,
        x_key="X_features",
        y_key="Y_features",
        read_half=False,         # lee como float16 para menos RAM/PCIe (por si acaso hay probleas de memoria)
        keep_device=None        # mueve a GPU en el loop
    )
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = 1
    val_size = dataset_size - train_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size ])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  num_workers=2, pin_memory=False, persistent_workers=False)


    #########################################################
    ###      iii) MODEL DEFINITION AND TRAINING LOOP      ###   
    #########################################################
    model = build_model(device, use_pretrained=use_pretrained, checkpoint_path=str(checkpoints_path))
    D     = PatchDiscriminator2D(in_ch=9, base=64).to(device)
    optG = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-6)
    optD = optim.Adam(D.parameters(),     lr=1e-4, betas=(0.5, 0.999))
    
    
    # Only ControlNet
    '''schedulerG = OneCycleLR(optG, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=num_epochs)
    fit(model, D, train_loader, val_loader, optG, optD, schedulerG, device, guidance_scale, num_epochs=num_epochs, output_dir=checkpoints_dir, lambda_adv=1e-3)'''

    # ControlNet+AnimateDiff
    schedulerG = OneCycleLR(optG, max_lr=1e-4,steps_per_epoch=len(train_loader),epochs=num_epochs)
    fit(model, D, train_loader, val_loader, optG, optD, schedulerG,device, guidance_scale, num_epochs, checkpoints_dir,lambda_adv=1e-3, L=L, stride=stride)


    model = build_model(device, use_pretrained = use_pretrained, checkpoint_path=str(checkpoints_path))

    
    model.eval()
    print('ALL DONE')
