import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score


class ConvINReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvINReLU(in_ch, out_ch),
            ConvINReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Correct channel bookkeeping:
      x:    [B, x_ch,  ...]
      skip: [B, skip_ch, ...]
    upsample maps x_ch -> out_ch, concat => (out_ch + skip_ch) -> out_ch
    """
    def __init__(self, x_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(x_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # pad if needed for odd sizes
        dz = skip.size(-3) - x.size(-3)
        dy = skip.size(-2) - x.size(-2)
        dx = skip.size(-1) - x.size(-1)
        if dz != 0 or dy != 0 or dx != 0:
            x = F.pad(
                x,
                [dx // 2, dx - dx // 2,
                 dy // 2, dy - dy // 2,
                 dz // 2, dz - dz // 2]
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3DFeatureTaps(nn.Module):
    def __init__(self, in_channels: int, base_ch: int = 32, seg_out_channels: int = 1, depth: int = 4):
        super().__init__()
        assert depth >= 3

        chs = [base_ch * (2 ** i) for i in range(depth)]  # e.g. [32,64,128,256]

        self.inc = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList([Down(chs[i], chs[i + 1]) for i in range(depth - 1)])

        # decoder: each up i maps from chs[depth-1-i] -> chs[depth-2-i]
        # and fuses with skip of channels chs[depth-2-i]
        self.ups = nn.ModuleList()
        for i in range(depth - 1):
            x_ch = chs[depth - 1 - i]
            out_ch = chs[depth - 2 - i]
            skip_ch = out_ch
            self.ups.append(Up(x_ch=x_ch, skip_ch=skip_ch, out_ch=out_ch))

        self.outc = nn.Conv3d(chs[0], seg_out_channels, kernel_size=1)

        self._enc_mid_idx = (depth - 2) // 2
        self._dec_mid_idx = max(0, (depth - 2) // 2)

        self._chs = chs  # keep for later reference if needed

    def forward(self, x):
        skips = []

        x0 = self.inc(x)
        skips.append(x0)
        enc_feats = [x0]

        x = x0
        for d in self.downs:
            x = d(x)
            skips.append(x)
            enc_feats.append(x)

        f_bot = x

        dec_feats = []
        x = f_bot
        for i, up in enumerate(self.ups):
            skip = skips[-2 - i]
            x = up(x, skip)
            dec_feats.append(x)

        seg_logits = self.outc(x)

        f_enc_mid = enc_feats[1 + self._enc_mid_idx]
        f_dec_mid = dec_feats[self._dec_mid_idx]

        return seg_logits, f_enc_mid, f_bot, f_dec_mid


class DepthwiseCompare(nn.Module):
    """
    Learns a per-channel comparison between (f_pre, f_post) using grouped 1x1x1 conv:

      in:  cat([f_pre, f_post], C=2C)
      out: C channels, groups=C  => each output channel sees ONLY its (pre, post) pair.

    This is a clean Torch interpretation of "depth-wise convolution" for pairwise comparison.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.cmp = nn.Conv3d(
            in_channels=2 * channels,
            out_channels=channels,
            kernel_size=1,
            groups=channels,
            bias=True
        )

    def forward(self, f_pre, f_post):
        x = torch.cat([f_pre, f_post], dim=1)
        return self.cmp(x)


class RPNet3D(nn.Module):
    """
    Main network:
      - shared-weight 3D U-Net applied to pre and post images
      - segmentation logits for both
      - response prediction head from 3 feature layers using depth-wise comparisons
    """
    def __init__(
        self,
        in_channels: int = 1,
        seg_out_channels: int = 1,
        num_classes: int = 2,
        base_ch: int = 8,
        depth: int = 4,
        cls_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.unet = UNet3DFeatureTaps(
            in_channels=in_channels,
            base_ch=base_ch,
            seg_out_channels=seg_out_channels,
            depth=depth
        )

        # We need the channel counts of the three taps to build DepthwiseCompare modules.
        # With the chosen U-Net, these are:
        chs = [base_ch * (2 ** i) for i in range(depth)]
        enc_mid_ch = chs[1 + (depth - 2) // 2]
        bot_ch = chs[-1]
        dec_mid_ch = chs[max(0, (depth - 2) // 2)]  # decoder out channels at that tap

        self.cmp1 = DepthwiseCompare(enc_mid_ch)
        self.cmp2 = DepthwiseCompare(bot_ch)
        self.cmp3 = DepthwiseCompare(dec_mid_ch)

        self.pool = nn.AdaptiveAvgPool3d(1)
        fused_dim = enc_mid_ch + bot_ch + dec_mid_ch

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(cls_hidden, num_classes),
        )

    def forward(self, x_pre, x_post):
        # shared U-Net
        seg_pre, f1_pre, f2_pre, f3_pre = self.unet(x_pre)
        seg_post, f1_post, f2_post, f3_post = self.unet(x_post)

        # depth-wise comparisons at 3 scales
        g1 = self.cmp1(f1_pre, f1_post)
        g2 = self.cmp2(f2_pre, f2_post)
        g3 = self.cmp3(f3_pre, f3_post)

        # global pooling + concat
        v1 = self.pool(g1).flatten(1)
        v2 = self.pool(g2).flatten(1)
        v3 = self.pool(g3).flatten(1)
        v = torch.cat([v1, v2, v3], dim=1)

        cls_logits = self.classifier(v)

        return {
            "seg_pre": seg_pre,        # [B, seg_out_channels, D, H, W]
            "seg_post": seg_post,      # [B, seg_out_channels, D, H, W]
            "cls_logits": cls_logits,  # [B, num_classes]
        }


# -------------------------
# Dice (binary) on logits
# -------------------------
def soft_dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    logits: [B, 1, D, H, W]
    targets: [B, 1, D, H, W] in {0,1}
    """
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.float().flatten(1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


# ---------------------------------------
# Box-counting fractal dimension (binary)
# ---------------------------------------
@torch.no_grad()
def fractal_dimension_boxcount(mask: torch.Tensor, min_box: int = 2, max_scales: int = 5, eps: float = 1e-6) -> torch.Tensor:
    """
    mask: [B, 1, D, H, W] binary {0,1}
    Returns: [B] fractal dimension estimate via box-counting.
    Minimal, robust-enough implementation (runs on GPU but uses pooling).
    """
    # Ensure binary float
    x = (mask > 0.5).float()

    B = x.shape[0]
    Ds = []
    Ns = []

    # choose box sizes as powers of 2: s = 2,4,8,... up to max_scales
    # we use max pool to detect if any voxel in a box is occupied
    for i in range(max_scales):
        s = min_box * (2 ** i)
        # stop if box larger than smallest dimension
        if s > min(x.shape[-3:]):
            break
        pooled = F.max_pool3d(x, kernel_size=s, stride=s, padding=0)
        # number of non-empty boxes
        n = (pooled > 0).flatten(1).sum(dim=1).float() + eps
        Ns.append(n)
        Ds.append(torch.full((B,), float(s), device=x.device))

    if len(Ns) < 2:
        # not enough scales: return 0 (or 1) safely
        return torch.zeros((B,), device=x.device)

    # Fit slope of log(N(s)) vs log(1/s) => FD = slope
    # x-axis: log(1/s) = -log(s)
    logs = torch.stack([torch.log(d) for d in Ds], dim=1)          # [B, K]
    logN = torch.stack([torch.log(n) for n in Ns], dim=1)          # [B, K]
    x_axis = -logs                                                 # [B, K]

    x_mean = x_axis.mean(dim=1, keepdim=True)
    y_mean = logN.mean(dim=1, keepdim=True)
    num = ((x_axis - x_mean) * (logN - y_mean)).sum(dim=1)
    den = ((x_axis - x_mean) ** 2).sum(dim=1) + eps
    fd = num / den
    return fd


def seg_loss_dice_plus_fd(seg_logits: torch.Tensor, seg_gt: torch.Tensor) -> torch.Tensor:
    """
    seg_logits: [B,1,D,H,W]
    seg_gt:     [B,1,D,H,W] {0,1}
    Implements: Dice + |FD(pred_bin) - FD(gt)|
    """
    dice = soft_dice_loss_with_logits(seg_logits, seg_gt)
    pred_bin = (torch.sigmoid(seg_logits) > 0.5).float()
    fd_pred = fractal_dimension_boxcount(pred_bin)
    fd_gt = fractal_dimension_boxcount(seg_gt)
    fd_term = (fd_pred - fd_gt).abs().mean()
    return dice + fd_term


# -------------------------
# Binary focal loss (logits)
# -------------------------
class BinaryFocalLoss(nn.Module):
    """
    Focal loss as described in supplement:
      -alpha (1-pt)^gamma log(pt)
    for binary labels y in {0,1}.
    We compute using logits for numerical stability.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B] or [B,1] or [B,2]
          - if [B,2], assumes logits for class1 at index 1.
        targets: [B] long/int {0,1}
        """
        targets = targets.long().view(-1)

        if logits.dim() == 2 and logits.size(1) == 2:
            # use class-1 logit minus class-0 logit => equivalent to binary logit
            logit = logits[:, 1] - logits[:, 0]
        else:
            logit = logits.view(-1)

        # BCE with logits gives: loss = -[y log(p) + (1-y) log(1-p)]
        bce = F.binary_cross_entropy_with_logits(logit, targets.float(), reduction="none")
        p = torch.sigmoid(logit)
        pt = torch.where(targets == 1, p, 1 - p)

        alpha_t = torch.where(targets == 1, torch.tensor(self.alpha, device=logit.device),
                              torch.tensor(1 - self.alpha, device=logit.device))
        loss = alpha_t * (1 - pt).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def read_yaml_file(path: str) -> dict:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class RPNetLightning(pl.LightningModule):
    def __init__(
        self,
        config_dir: str,
        target_key: str,
        w_cls: float = 1.0,
        w_seg_pre: float = 0.2,
        w_seg_post: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = RPNet3D()
        self.target_key = target_key

        # same config convention as your code
        config = read_yaml_file(config_dir)
        self.lr = config["lr"]
        self.max_epochs = config["max_epochs"]

        # losses
        self.cls_criterion = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        # seg uses function seg_loss_dice_plus_fd

        # weights (paper used w_cls=1 and w_seg=0.2 for each seg branch) :contentReference[oaicite:1]{index=1}
        self.w_cls = w_cls
        self.w_seg_pre = w_seg_pre
        self.w_seg_post = w_seg_post

        # metrics
        self.acc = BinaryAccuracy()
        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()

    def forward(self, batch):
        x_pre = batch["base_img"].to(self.device)    # [B,C,D,H,W]
        x_post = batch["followup_img"].to(self.device)  # [B,C,D,H,W]
        return self.model(x_pre, x_post)

    def _shared_step(self, batch, stage: str):
        out = self(batch)

        # ----- classification -----
        y = batch["targets"][self.target_key].long().to(self.device)  # [B] {0,1}
        cls_logits = out["cls_logits"]  # [B,2] or [B]
        cls_loss = self.cls_criterion(cls_logits, y)

        # prob for metrics
        if cls_logits.dim() == 2 and cls_logits.size(1) == 2:
            prob1 = torch.softmax(cls_logits, dim=1)[:, 1]
            y_hat = torch.argmax(cls_logits, dim=1)
        else:
            prob1 = torch.sigmoid(cls_logits.view(-1))
            y_hat = (prob1 > 0.5).long()

        # ----- segmentation -----
        # seg labels expected [B,1,D,H,W] in {0,1}
        seg_pre_gt = batch["base_seg"].to(self.device)
        seg_post_gt = batch["followup_seg"].to(self.device)

        seg_pre_logits = out["seg_pre"]
        seg_post_logits = out["seg_post"]

        seg_pre_loss = seg_loss_dice_plus_fd(seg_pre_logits, seg_pre_gt)
        seg_post_loss = seg_loss_dice_plus_fd(seg_post_logits, seg_post_gt)

        loss = self.w_cls * cls_loss + self.w_seg_pre * seg_pre_loss + self.w_seg_post * seg_post_loss

        # logging
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_cls_loss", cls_loss, prog_bar=False, on_epoch=True)
        self.log(f"{stage}_seg_pre_loss", seg_pre_loss, prog_bar=False, on_epoch=True)
        self.log(f"{stage}_seg_post_loss", seg_post_loss, prog_bar=False, on_epoch=True)

        self.log(f"{stage}_acc", self.acc(y_hat, y), prog_bar=False)
        self.log(f"{stage}_auroc", self.auroc(prob1, y), prog_bar=True)
        self.log(f"{stage}_f1", self.f1(y_hat, y), prog_bar=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # same linear decay scheduler you used
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (self.lr / self.max_epochs) * (self.max_epochs - epoch)
            if epoch < self.max_epochs else 0.0
        )
        return [optimizer], [scheduler]


# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    B, C, D, H, W = 2, 4, 64, 128, 128  # e.g., 4 MRI sequences as channels
    x_pre = torch.randn(B, C, D, H, W)
    x_post = torch.randn(B, C, D, H, W)

    model = RPNet3D(in_channels=C, seg_out_channels=1, num_classes=2, base_ch=16, depth=4)
    out = model(x_pre, x_post)
    print(out["seg_pre"].shape, out["seg_post"].shape, out["cls_logits"].shape)