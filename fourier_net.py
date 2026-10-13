"""Learnable Fourier front end for ship detection from CPHD phase history (train_raw.py --arch fourier).

Why: after a crop is re-referenced to its centre, a scatterer at ground position r adds exp(+j 2 pi k(p,n) . r) to
sample (pulse p, frequency n), where k is the ground wavenumber of that sample. Position is the *frequency* of a phase
ramp across the whole array, which a ViT on patches could not learn (it memorised, but did not generalise). Here the
network starts from the transform that turns those ramps into local peaks, and keeps it learnable:

    X (P x S complex)  ->  taper (learnable, separable)
                       ->  Z = A_p  X  A_s^T        A_s, A_p: learnable complex P x P, S x S, initialised to the DFT
                       ->  also Z1, Z2 from the first / second half of the pulses (two sub-apertures)
    features: log|Z|^2, log|Z1|^2, log|Z2|^2, and the sub-aperture coherence Z1 conj(Z2) / |Z1||Z2| (real, imag)
              (phase information a detected GEC does not have: motion, coherence of hard targets vs sea)
    -> ResNet-18 stem to stride 4 (shift-equivariant: in this space a ship is local and moves when the ship moves)
    -> features sampled onto the 4 m east/north cell grid at each cell's exact transform bin, from the crop's own
       geometry: bin = N/2 + N (dk . r), dk = change of the ground wavenumber per sample along each axis
    -> the same heat / offset / size heads and losses as the ViT model

The transform bins are about 1 m (the crops are built for 1 m range and cross-range resolution), so the stride-4
features are about 4 m, matching the 4 m output cells.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

C = 299792458.0


class LearnableDFT(nn.Module):
    """Complex N x N matrix (bins x samples) initialised to the centred DFT: bin b <-> (b - N/2) / N cycles per sample."""
    def __init__(self, n, learn=True):
        super().__init__()
        k = torch.arange(n, dtype=torch.float64)
        b = torch.arange(n, dtype=torch.float64) - n // 2
        ang = -2 * math.pi * b[:, None] * k[None, :] / n
        s = 1.0 / math.sqrt(n)
        self.wr = nn.Parameter((torch.cos(ang) * s).float(), requires_grad=learn)
        self.wi = nn.Parameter((torch.sin(ang) * s).float(), requires_grad=learn)

    def weight(self):
        return torch.complex(self.wr, self.wi)


class FourierShipNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import torchvision
        self.cfg = cfg
        P, S = cfg["pulses"], cfg["samples"]
        learn = not cfg.get("fixed_dft", False)
        self.dft_p, self.dft_s = LearnableDFT(P, learn), LearnableDFT(S, learn)
        self.taper_p = nn.Parameter(torch.hann_window(P, periodic=False) * 0.9 + 0.1, requires_grad=learn)
        self.taper_s = nn.Parameter(torch.hann_window(S, periodic=False) * 0.9 + 0.1, requires_grad=learn)
        r18 = torchvision.models.resnet18(weights="DEFAULT" if cfg.get("pretrained", True) else None)
        w = r18.conv1.weight.data.mean(1, keepdim=True)                # RGB filters -> 5 input channels
        self.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        self.conv1.weight.data = w.repeat(1, 5, 1, 1) * (3 / 5)
        self.bn1, self.relu, self.maxpool = r18.bn1, r18.relu, r18.maxpool
        self.layer1, self.layer2 = r18.layer1, r18.layer2              # stride 4 (64 ch), stride 8 (128 ch)
        dq = cfg["dec_dim"]
        self.fuse = nn.Sequential(nn.Conv2d(64 + 128, dq, 1), nn.GroupNorm(8, dq), nn.ReLU(inplace=True))
        conv = lambda: nn.Sequential(nn.Conv2d(dq, dq, 3, padding=1), nn.GroupNorm(8, dq), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(conv(), conv())
        self.heat = nn.Conv2d(dq, 1, 1)
        self.off = nn.Conv2d(dq, 2, 1)
        self.size = nn.Conv2d(dq, 2, 1)
        nn.init.constant_(self.heat.bias, -2.19)
        nn.init.constant_(self.size.bias, math.log(40.0))
        cells, cm = cfg["cells"], cfg["cell_m"]
        H = cells * cm / 2
        c = (torch.arange(cells, dtype=torch.float64) + 0.5) * cm
        n_, e_ = torch.meshgrid(H - c, c - H, indexing="ij")           # row 0 = north edge, as in the labels
        self.register_buffer("cell_en", torch.stack([e_, n_], -1), persistent=False)   # (cells, cells, 2) metres

    def front_parameters(self):
        return [self.dft_p.wr, self.dft_p.wi, self.dft_s.wr, self.dft_s.wi, self.taper_p, self.taper_s]

    # -------------------------------------------------------------- transform
    def transform(self, X):
        """X (B, P, S) complex -> full aperture Z and sub-apertures Z1, Z2, each (B, P, S) in transform bins."""
        P = X.shape[1]
        X = X * (self.taper_p[:, None] * self.taper_s[None, :]).to(X.dtype)
        Y = X @ self.dft_s.weight().T.to(X.dtype)                       # along frequency: range
        Wp = self.dft_p.weight().to(X.dtype)
        h = P // 2
        Z1 = Wp[:, :h] @ Y[:, :h]                                       # along pulses: cross-range, per sub-aperture
        Z2 = Wp[:, h:] @ Y[:, h:]
        return Z1 + Z2, Z1, Z2

    @staticmethod
    def features(Z, Z1, Z2, eps=1e-6):
        def logp(z):
            p = torch.log(z.real ** 2 + z.imag ** 2 + eps)
            return p - p.mean(dim=(1, 2), keepdim=True)
        coh = Z1 * Z2.conj() / (Z1.abs() * Z2.abs() + eps)
        return torch.stack([logp(Z) / 4, logp(Z1) / 4, logp(Z2) / 4, coh.real, coh.imag], 1)

    def sample_grid(self, b):
        """Normalised (x, y) of every output cell in the transform image, from the crop's geometry."""
        X = b["X"]
        B, P, S = X.shape
        u = b["u"].double()                                             # (B, P, 3) crop centre -> antenna, unit
        E, N = b["E"].double(), b["N"].double()
        uE, uN = (u * E[:, None, :]).sum(-1), (u * N[:, None, :]).sum(-1)               # (B, P)
        f0, scss = b["f0"].double(), b["scss"].double()
        pm, nm = P // 2, S // 2
        dk_f = torch.stack([uE[:, pm], uN[:, pm]], -1) * (2 / C) * scss[:, None]       # per frequency sample
        f_mid = f0 + nm * scss[:, None]                                                 # (B, P)
        k_mid = torch.stack([uE, uN], -1) * (2 / C) * f_mid[..., None]                  # (B, P, 2)
        dk_p = (k_mid[:, -1] - k_mid[:, 0]) / (P - 1)                                   # per pulse
        r = self.cell_en.double()                                                        # (cells, cells, 2)
        b_rg = S / 2 + S * torch.einsum("hwc,bc->bhw", r, dk_f)
        b_az = P / 2 + P * torch.einsum("hwc,bc->bhw", r, dk_p)
        gx = (b_rg + 0.5) / S * 2 - 1                                                    # columns = range bins
        gy = (b_az + 0.5) / P * 2 - 1                                                    # rows = cross-range bins
        return torch.stack([gx, gy], -1).float()

    def forward(self, b):
        with torch.autocast("cuda", enabled=False):
            Z, Z1, Z2 = self.transform(b["X"].to(torch.complex64))
            x = self.features(Z, Z1, Z2).float()
            grid = self.sample_grid(b)
        amp = self.cfg["amp"] and x.is_cuda
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else _null()
        with ctx:
            y = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            f4 = self.layer1(y)
            f8 = self.layer2(f4)
            f = self.fuse(torch.cat([f4, F.interpolate(f8, size=f4.shape[-2:], mode="bilinear", align_corners=False)], 1))
            g = F.grid_sample(f.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            y = self.trunk(g)
            return dict(heat=self.heat(y).float(), off=self.off(y).float(), size=self.size(y).float())


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
