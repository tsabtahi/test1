"""Standard object detector on the Fourier features of the CPHD crop (train_raw.py --arch det).

    X (P x S complex phase history)
      -> the Fourier front end of fourier_net.py (taper, DFT along range and cross-range, two half-apertures)
      -> feature channels per transform bin:  logZ  logZ1  logZ2  coh_re  coh_im   (pick with --det-channels)
      -> resampled onto a north-up east/north image, --det-pixel-m per pixel (default 1 m: 512 x 512 for 512 m),
         from the crop's own geometry, so the image is in the same frame as the labels
      -> a torchvision detector: Faster R-CNN (default), RetinaNet or FCOS, with a ResNet-FPN backbone whose first
         conv takes the chosen channels (ImageNet weights, RGB filters averaged over the new channels)

No ResNet-18 stem or heat/offset/size heads: the detector's own anchors, losses and NMS. Ignored ships (cut off by
the crop edge, or crowd) are left out of the training targets. Output boxes are converted back to crop-local metres
so train_raw.py's evaluation, collages and GeoJSON work unchanged.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fourier_net import FourierShipNet, LearnableDFT

CHANNELS = ("logZ", "logZ1", "logZ2", "coh_re", "coh_im")


class _Front(nn.Module):
    """The Fourier front end only (same maths and parameter names as FourierShipNet), sampled onto any E/N grid."""
    transform = FourierShipNet.transform
    features = staticmethod(FourierShipNet.features)
    sample_grid = FourierShipNet.sample_grid                      # reads self.cell_en: (n, n, 2) east/north metres

    def __init__(self, P, S, n_pix, pixel_m, learn):
        super().__init__()
        self.dft_p, self.dft_s = LearnableDFT(P, learn), LearnableDFT(S, learn)
        self.taper_p = nn.Parameter(torch.hann_window(P, periodic=False) * 0.9 + 0.1, requires_grad=learn)
        self.taper_s = nn.Parameter(torch.hann_window(S, periodic=False) * 0.9 + 0.1, requires_grad=learn)
        H = n_pix * pixel_m / 2
        c = (torch.arange(n_pix, dtype=torch.float64) + 0.5) * pixel_m
        n_, e_ = torch.meshgrid(H - c, c - H, indexing="ij")         # row 0 = north edge
        self.register_buffer("cell_en", torch.stack([e_, n_], -1), persistent=False)


def _backbone(name, in_ch, pretrained, retina_like):
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.ops.feature_pyramid_network import LastLevelP6P7
    import torchvision
    weights = None
    if pretrained:
        weights = {"resnet18": torchvision.models.ResNet18_Weights.DEFAULT,      # the ones cached in the image
                   "resnet34": torchvision.models.ResNet34_Weights.DEFAULT,
                   "resnet50": torchvision.models.ResNet50_Weights.DEFAULT}[name]
    kw = dict(returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)) if retina_like else {}
    try:
        bb = resnet_fpn_backbone(backbone_name=name, weights=weights, trainable_layers=5, **kw)
    except Exception as e:                                       # offline / unwritable cache: random init
        print(f"  {name} ImageNet weights not available ({type(e).__name__}: {e}); starting from random weights")
        bb = resnet_fpn_backbone(backbone_name=name, weights=None, trainable_layers=5, **kw)
    old = bb.body.conv1
    new = nn.Conv2d(in_ch, old.out_channels, 7, 2, 3, bias=False)
    new.weight.data = old.weight.data.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * (3 / in_ch)
    bb.body.conv1 = new
    return bb


class DetShipNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from torchvision.models.detection import FasterRCNN, RetinaNet, FCOS
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        self.cfg = cfg
        self.px = float(cfg.get("det_pixel_m", 1.0))
        self.H = cfg["cells"] * cfg["cell_m"] / 2
        self.n = int(round(2 * self.H / self.px))
        self.ch = [CHANNELS.index(c) for c in cfg.get("det_channels") or CHANNELS]
        self.front = _Front(cfg["pulses"], cfg["samples"], self.n, self.px, not cfg.get("fixed_dft", False))
        kind, bname = cfg.get("detector", "fasterrcnn"), cfg.get("det_backbone", "resnet50")
        retina_like = kind in ("retinanet", "fcos")
        bb = _backbone(bname, len(self.ch), cfg.get("pretrained", True), retina_like)
        # ships are 5-400 m long and lie at any heading (axis-aligned boxes): small anchors, wide aspect ratios
        s0 = max(4.0, 8.0 / self.px)
        sizes = tuple((s0 * 2 ** i,) for i in range(5))
        ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * 5
        common = dict(min_size=self.n, max_size=self.n, image_mean=[0.0] * len(self.ch), image_std=[1.0] * len(self.ch))
        if kind == "fasterrcnn":
            self.det = FasterRCNN(bb, num_classes=2, rpn_anchor_generator=AnchorGenerator(sizes, ratios),
                                  box_detections_per_img=cfg.get("max_objs", 100), box_score_thresh=0.001, **common)
        elif kind == "retinanet":
            self.det = RetinaNet(bb, num_classes=2, anchor_generator=AnchorGenerator(sizes, ratios),
                                 detections_per_img=cfg.get("max_objs", 100), score_thresh=0.001, **common)
        elif kind == "fcos":
            self.det = FCOS(bb, num_classes=2, anchor_generator=AnchorGenerator(sizes, ((1.0,),) * 5),
                            detections_per_img=cfg.get("max_objs", 100), score_thresh=0.001, **common)
        else:
            raise ValueError(f"unknown detector {kind}")

    def front_parameters(self):
        f = self.front
        return [f.dft_p.wr, f.dft_p.wi, f.dft_s.wr, f.dft_s.wi, f.taper_p, f.taper_s]

    def image(self, b):
        """(B, C, n, n) feature image, north up, det_pixel_m per pixel."""
        with torch.autocast("cuda", enabled=False):
            Z, Z1, Z2 = self.front.transform(b["X"].to(torch.complex64))
            x = self.front.features(Z, Z1, Z2).float()[:, self.ch]
            g = F.grid_sample(x, self.front.sample_grid(b), mode="bilinear", padding_mode="zeros", align_corners=False)
        return g

    # crop-local metres [e0, n0, e1, n1] <-> image pixels [x0, y0, x1, y1]
    def m2px(self, bx):
        return torch.stack([(bx[:, 0] + self.H) / self.px, (self.H - bx[:, 3]) / self.px,
                            (bx[:, 2] + self.H) / self.px, (self.H - bx[:, 1]) / self.px], 1)

    def px2m(self, bx):
        return torch.stack([bx[:, 0] * self.px - self.H, self.H - bx[:, 3] * self.px,
                            bx[:, 2] * self.px - self.H, self.H - bx[:, 1] * self.px], 1)

    def targets(self, b, device):
        out = []
        for gt, ig in zip(b["gt"], b["gt_ignore"]):
            gt = torch.as_tensor(gt, dtype=torch.float32, device=device).reshape(-1, 4)
            keep = ~torch.as_tensor(ig, dtype=torch.bool, device=device).reshape(-1)
            p = self.m2px(gt[keep]).clamp(0, self.n) if keep.any() else torch.zeros((0, 4), device=device)
            ok = ((p[:, 2] - p[:, 0]) >= 1) & ((p[:, 3] - p[:, 1]) >= 1)
            p = p[ok]
            out.append(dict(boxes=p, labels=torch.ones(len(p), dtype=torch.int64, device=device)))
        return out

    def forward(self, b):
        img = self.image(b)
        amp = self.cfg.get("amp") and img.is_cuda
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else _null()
        if self.training:
            with ctx:
                losses = self.det(list(img), self.targets(b, img.device))
            return dict(det_losses={k: v.float() for k, v in losses.items()})
        with ctx:
            res = self.det(list(img))
        dets = [(r["scores"].float(), self.px2m(r["boxes"].float())) for r in res]
        return dict(dets=dets, image=img)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
