import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# -------------------------
# Low-level blocks (Keras-like)
# -------------------------
class ConvINLReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ContextModule(nn.Module):
    """
    Keras:
      convolution1 = conv_block(...)
      dropout = SpatialDropout3D(...)
      convolution2 = conv_block(...)
    """
    def __init__(self, ch, dropout_rate=0.3):
        super().__init__()
        self.conv1 = ConvINLReLU(ch, ch, k=3, stride=1, padding=1)
        self.drop = nn.Dropout3d(p=dropout_rate)  # Dropout3d ~ SpatialDropout3D
        self.conv2 = ConvINLReLU(ch, ch, k=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv2(self.drop(self.conv1(x)))


class LocalizationModule(nn.Module):
    """
    Keras:
      convolution1 = conv_block(..., n_filters)
      convolution2 = conv_block(convolution1, n_filters, kernel=(1,1,1))
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvINLReLU(in_ch, out_ch, k=3, stride=1, padding=1)
        self.conv2 = ConvINLReLU(out_ch, out_ch, k=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class UpSamplingModule(nn.Module):
    """
    Keras:
      up = UpSampling3D(...)
      conv = conv_block(up, n_filters)
    """
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode="nearest")
        self.conv = ConvINLReLU(in_ch, out_ch, k=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


# -------------------------
# Backbone: siam3dunet_backbone in Torch
# -------------------------
class Siam3DUNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_base_filters: int = 16,
        depth: int = 5,
        dropout_rate: float = 0.3,
        n_segmentation_levels: int = 3,
        n_labels: int = 4,
        activation_name: str = "sigmoid",  # used for final mask
    ):
        super().__init__()
        assert depth >= 3
        assert n_segmentation_levels <= depth - 1

        self.depth = depth
        self.n_segmentation_levels = n_segmentation_levels
        self.n_labels = n_labels
        self.activation_name = activation_name

        # encoder per-level blocks
        self.in_convs = nn.ModuleList()
        self.contexts = nn.ModuleList()
        self.level_filters = []

        prev_ch = in_channels
        for level in range(depth):
            ch = (2 ** level) * n_base_filters
            self.level_filters.append(ch)
            stride = 1 if level == 0 else 2
            self.in_convs.append(ConvINLReLU(prev_ch, ch, k=3, stride=stride, padding=1))
            self.contexts.append(ContextModule(ch, dropout_rate=dropout_rate))
            prev_ch = ch

        # decoder
        self.ups = nn.ModuleList()
        self.locals = nn.ModuleList()
        for level in range(depth - 2, -1, -1):
            # up from current ch -> level ch
            self.ups.append(UpSamplingModule(self.level_filters[level + 1], self.level_filters[level], scale=2))
            # concat(skip, up) => 2*ch -> ch
            self.locals.append(LocalizationModule(in_ch=2 * self.level_filters[level], out_ch=self.level_filters[level]))

        # deep supervision seg heads for levels < n_segmentation_levels
        # Keras creates seg layers for levels: 0,1,2 (if n_segmentation_levels=3)
        self.seg_heads = nn.ModuleList([
            nn.Conv3d(self.level_filters[level], n_labels, kernel_size=1)
            for level in range(n_segmentation_levels)
        ])

    def _activate_mask(self, x):
        if self.activation_name == "sigmoid":
            return torch.sigmoid(x)
        if self.activation_name == "softmax":
            return torch.softmax(x, dim=1)
        raise ValueError(f"Unsupported activation_name={self.activation_name}")

    def forward(self, x):
        return_layers = []

        # encoder
        level_outputs = []
        cur = x
        for level in range(self.depth):
            in_conv = self.in_convs[level](cur)
            ctx = self.contexts[level](in_conv)
            summation = in_conv + ctx  # residual Add
            level_outputs.append(summation)
            cur = summation

            if level in (2, 4):  # Keras taps
                return_layers.append(cur)

        # decoder + deep supervision accumulation
        seg_logits_levels = [None] * self.n_segmentation_levels

        # iterate decoder levels from depth-2 down to 0
        for dec_i, level in enumerate(range(self.depth - 2, -1, -1)):
            up = self.ups[dec_i](cur)
            # concat along channel dim
            cat = torch.cat([level_outputs[level], up], dim=1)
            cur = self.locals[dec_i](cat)

            # Keras: if level_number < n_segmentation_levels: store Conv3D(n_labels,1) for that level
            if level < self.n_segmentation_levels:
                seg_logits_levels[level] = self.seg_heads[level](cur)

        # Keras: output_layer = sum of seg_logits at seg levels, upsample between sums
        output_layer = None
        for level in reversed(range(self.n_segmentation_levels)):
            seg_l = seg_logits_levels[level]
            if seg_l is None:
                raise RuntimeError("seg_logits_levels missing a level; check indexing.")
            output_layer = seg_l if output_layer is None else (output_layer + seg_l)

            # Keras: if level_number == 1: return_layers.append(output_layer)
            if level == 1:
                return_layers.append(output_layer)

            if level > 0:
                output_layer = F.interpolate(output_layer, scale_factor=2, mode="nearest")

        mask = self._activate_mask(output_layer)
        return_layers.append(mask)

        # return_layers indices match Keras usage:
        # [0]=enc level2 feat, [1]=enc level4 feat, [2]=decoder aggregated seg logits at level1, [3]=final mask
        return return_layers


# -------------------------
# Full Siamese network (matches keras siam3dunet_model)
# -------------------------
class Siam3DUNetTorch(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,         # Keras input_shape=(4,128,128,128)
        num_classes: int = 2,            # Keras Dense(1) with sigmoid => 2 classes (binary)
        n_labels: int = 1,            # Keras n_labels=4
        n_base_filters: int = 16,
        depth: int = 5,
        dropout_rate: float = 0.3,
        n_segmentation_levels: int = 3,
        activation_name: str = "sigmoid",
    ):
        super().__init__()

        self.backbone = Siam3DUNetBackbone(
            in_channels=in_channels,
            n_base_filters=n_base_filters,
            depth=depth,
            dropout_rate=dropout_rate,
            n_segmentation_levels=n_segmentation_levels,
            n_labels=n_labels,
            activation_name=activation_name,
        )

        # Keras does: Subtract -> create_convolution_block(..., 32) at each tap
        self.sf_conv0 = ConvINLReLU((2 ** 2) * n_base_filters, 32, k=3, stride=1, padding=1)  # level 2 filters
        self.sf_conv1 = ConvINLReLU((2 ** 4) * n_base_filters, 32, k=3, stride=1, padding=1)  # level 4 filters
        self.sf_conv2 = ConvINLReLU(n_labels, 32, k=3, stride=1, padding=1)                   # seg logits channels

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.score_fc = nn.Linear(32 * 3, num_classes)  # Dense(1)
        self.activation_name = activation_name

    def _activate_score(self, x):
        # Keras uses Activation(activation_name) on the score too (usually sigmoid)
        if self.activation_name == "sigmoid":
            return torch.sigmoid(x)
        if self.activation_name == "softmax":
            # softmax doesn't make sense for scalar; keep sigmoid behavior if user set softmax for masks
            return torch.sigmoid(x)
        return torch.sigmoid(x)

    def forward(self, x1, x2):
        r1 = self.backbone(x1)
        r2 = self.backbone(x2)

        sf1_0, sf1_1, sf1_2 = r1[0], r1[1], r1[2]  # taps
        sf2_0, sf2_1, sf2_2 = r2[0], r2[1], r2[2]

        # Keras: Subtract then conv to 32
        g0 = self.sf_conv0(sf1_0 - sf2_0)
        g1 = self.sf_conv1(sf1_1 - sf2_1)
        g2 = self.sf_conv2(sf1_2 - sf2_2)

        v0 = self.pool(g0).flatten(1)
        v1 = self.pool(g1).flatten(1)
        v2 = self.pool(g2).flatten(1)

        v = torch.cat([v0, v1, v2], dim=1)
        score_logits = self.score_fc(v)
        # score = self._activate_score(score_logits)

        mask1 = r1[-1]  # activated mask
        mask2 = r2[-1]

        return score_logits, mask1, mask2


def read_yaml_file(path: str) -> dict:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def soft_dice_loss(probs, target, eps=1e-6):
    # probs, target: [B, C, D, H, W] (or [B,1,D,H,W]) with probs in [0,1]
    probs = probs.float()
    target = target.float()

    dims = tuple(range(2, probs.dim()))
    intersection = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)

    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()

class RPNetLightning(pl.LightningModule):
    def __init__(
        self,
        config_dir: str,
        target_key: str,
        w_cls: float = 1.0,
        w_seg_pre: float = 0.2,
        w_seg_post: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()


        self.target_key = target_key

        config = read_yaml_file(config_dir)
        self.lr = config["lr"]
        self.max_epochs = config["max_epochs"]
        self.n_base_filters = config["n_base_filters"]

        self.model = Siam3DUNetTorch(n_base_filters=self.n_base_filters)
        # Keras: binary_crossentropy on sigmoid(score)
        self.cls_criterion = nn.CrossEntropyLoss()

        self.w_cls = w_cls
        self.w_seg_pre = w_seg_pre
        self.w_seg_post = w_seg_post

        self.acc = BinaryAccuracy()
        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()

    def forward(self, batch):
        x_pre = batch["base_img"].to(self.device)        # [B,C,D,H,W]
        x_post = batch["followup_img"].to(self.device)   # [B,C,D,H,W]
        return self.model(x_pre, x_post) # (score, mask1, mask2)

    def _shared_step(self, batch, stage: str):
        score, mask_pre_pred, mask_post_pred = self(batch)  # <- CHANGED

        # ----- classification -----
        y = batch["targets"][self.target_key].to(self.device).long()
        cls_loss = self.cls_criterion(score, y)

        y_hat = torch.argmax(score, dim=1)  # [B] predicted class indices
        prob1 = torch.softmax(score, dim=1)[:, 1]  # [B] probability of class 1
        y_long = y.long()

        # ----- segmentation -----
        seg_pre_gt = batch["base_seg"].to(self.device).float()
        seg_post_gt = batch["followup_seg"].to(self.device).float()

        # Your defined model outputs ACTIVATED masks (sigmoid), so use dice on probs.
        # If your soft_dice_loss_with_logits expects logits, do NOT use it here.
        seg_pre_loss = soft_dice_loss(mask_pre_pred, seg_pre_gt)     # <- CHANGED
        seg_post_loss = soft_dice_loss(mask_post_pred, seg_post_gt)  # <- CHANGED

        loss = self.w_cls * cls_loss + self.w_seg_pre * seg_pre_loss + self.w_seg_post * seg_post_loss

        # logging
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_cls_loss", cls_loss, prog_bar=False, on_epoch=True)
        self.log(f"{stage}_seg_pre_loss", seg_pre_loss, prog_bar=False, on_epoch=True)
        self.log(f"{stage}_seg_post_loss", seg_post_loss, prog_bar=False, on_epoch=True)

        self.log(f"{stage}_acc", self.acc(y_hat, y_long), prog_bar=False)
        self.log(f"{stage}_auroc", self.auroc(prob1, y_long), prog_bar=True)
        self.log(f"{stage}_f1", self.f1(y_hat, y_long), prog_bar=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Keras uses SGD(lr, momentum=0.9)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)  # <- CHANGED

        # LambdaLR expects a MULTIPLIER, not an absolute LR
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: max(0.0, (self.max_epochs - epoch) / float(self.max_epochs))  # <- CHANGED
        )
        return [optimizer], [scheduler]

# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    B, C, D, H, W = 2, 1, 64, 128, 128  # e.g., 4 MRI sequences as channels
    x_pre = torch.randn(B, C, D, H, W)
    x_post = torch.randn(B, C, D, H, W)

    model = Siam3DUNetTorch(in_channels=C)
    out = model(x_pre, x_post)
    print(out[0].shape, out[1].shape, out[2].shape)
    print(f"out0 (score) range: {out[0].min().item()} to {out[0].max().item()}")
    print(f"out1 (mask1) range: {out[1].min().item()} to {out[1].max().item()}")
    print(f"out2 (mask2) range: {out[2].min().item()} to {out[2].max().item()}")