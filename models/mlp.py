import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score


def read_yaml_file(file_path):
    import yaml
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            # nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x):
        return self.classifier(x)


class StatisticalPoolingMLP(pl.LightningModule):
    def __init__(self, features_dim: int, demographic_dim: int, config_dir: str, target_key: str ):
        super(StatisticalPoolingMLP, self).__init__()
        self.save_hyperparameters()
        self.target_key = target_key
        
        input_dim = 2 * 4 * features_dim  + demographic_dim # statistical pooling: mean, max, min, median
        dim = 256 #int((4/5)*features_dim)
        
        self.classifier_head = Classifier(input_dim=input_dim, hidden_dim=dim, output_dim=2)

        config = read_yaml_file(config_dir)
        self.lr = config['lr']
        self.max_epochs = config['max_epochs']  

        class_weights = torch.tensor([0.63, 1.37], dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights) 

        # Metrics
        self.acc = BinaryAccuracy()
        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()

    def _statistical_pooling(self, x: torch.Tensor, pad_mask: torch.Tensor):
        """
        x:        [B, T, F]
        pad_mask: [B, T]  True = padded
        returns:  [B, 4F] concat(mean, max, min, median)
        """
        B, T, F = x.shape
        device = x.device
        eps = 1e-8

        # counts
        valid = ~pad_mask                               # [B, T]
        count = valid.sum(dim=1, keepdim=True)          # [B, 1]

        # ---- mean (ignore pads) ----
        x0 = x.masked_fill(pad_mask.unsqueeze(-1), 0.0) # [B, T, F]
        mean = x0.sum(dim=1) / count.clamp_min(1)       # [B, F]
        mean = torch.where(count > 0, mean, mean.new_zeros(B, F))

        # ---- max/min (ignore pads) ----
        # use +/-inf so pads never win; then fix all-pad rows
        x_max = x.masked_fill(pad_mask.unsqueeze(-1), float("-inf"))
        x_min = x.masked_fill(pad_mask.unsqueeze(-1), float("inf"))

        maxv = x_max.max(dim=1).values                  # [B, F]
        minv = x_min.min(dim=1).values                  # [B, F]

        all_pad = (count == 0)                          # [B, 1]
        maxv = torch.where(all_pad, maxv.new_zeros(B, F), maxv)
        minv = torch.where(all_pad, minv.new_zeros(B, F), minv)

        # ---- std (ignore pads) ----
        # IMPORTANT: compute variance only over valid timesteps (pads contribute 0 weight)
        diff = x - mean.unsqueeze(1)                    # [B, T, F]
        diff2 = diff.square().masked_fill(pad_mask.unsqueeze(-1), 0.0)
        var = diff2.sum(dim=1) / count.clamp_min(1)     # [B, F]
        std = (var + eps).sqrt()
        std = torch.where(all_pad, std.new_zeros(B, F), std)

        # ---- median (ignore pads) ----
        # nanmedian is vectorized; requires float
        if not x.is_floating_point():
            x = x.float()
        x_nan = x.masked_fill(pad_mask.unsqueeze(-1), float("nan"))
        median = torch.nanmedian(x_nan, dim=1).values   # [B, F]
        median = torch.nan_to_num(median, nan=0.0)
        return torch.cat([mean, maxv, minv, median], dim=1)  # [B, 4F]


    def forward(self, batch):
        base_emb = self._statistical_pooling(x=batch["base"]["features"].to(self.device), pad_mask=batch["base"]["pad_mask"].to(self.device))
        followup_emb = self._statistical_pooling(x=batch["followup"]["features"].to(self.device), pad_mask=batch["followup"]["pad_mask"].to(self.device))
        combined_emb = torch.cat([base_emb, followup_emb, batch["demographic_info"].to(self.device)], dim=1)
        logits = self.classifier_head(combined_emb)
        return logits

    def _shared_step(self, batch, stage):
        logits = self(batch)
        gt_label = batch["targets"][self.target_key].long()
        loss = self.criterion(logits, gt_label)
        y_hat = torch.argmax(logits, dim=1)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc", self.acc(y_hat, gt_label), prog_bar=False)
        self.log(f"{stage}_auroc", self.auroc(y_hat, gt_label), prog_bar=True)
        self.log(f"{stage}_f1", self.f1(y_hat, gt_label), prog_bar=False)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
        # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (self.lr/self.max_epochs)*(self.max_epochs - epoch) if epoch < self.max_epochs else 0)
        # return [optimizer], [scheduler]