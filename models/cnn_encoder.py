import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
import pytorch_lightning as pl
from monai.networks.nets import Regressor, densenet
import torch.nn.functional as F


def read_yaml_file(file_path):
    import yaml
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

class CnnConcatenation(pl.LightningModule):
    def __init__(self, config_dir: str):
        super(CnnConcatenation, self).__init__()
        self.save_hyperparameters()
        config = read_yaml_file(config_dir)
        # self.model = Regressor(in_shape=(2, config["resize"][0], config["resize"][1], config["resize"][2]), out_shape=(2,), channels=(16, 32, 64, 128, 256, 512), strides=(2, 2, 2, 2, 2, 2))
        self.model = densenet.densenet121(spatial_dims=3, in_channels=2, out_channels=2)


        self.lr = config['lr']
        self.max_epochs = config['max_epochs']

        class_weights = torch.tensor([0.63, 1.37], dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)
        self.criterion_early_response = nn.CrossEntropyLoss(weight=self.class_weights)

        # Metrics
        self.acc = BinaryAccuracy()
        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()

    def forward(self, batch):
        base_img = batch["base_img"].to(self.device)
        followup_img = batch["followup_img"].to(self.device)
        # concatenate along channel dimension
        x = torch.cat([base_img, followup_img], dim=1)
        logits_early_response = self.model(x)
        logits = {"early_response": logits_early_response} #"survival": logits_overall_survival, 
        return logits

    def _shared_step(self, batch, stage):
        logits = self(batch)
        
        gt_early_response = batch["early_response"].long()
        loss = self.criterion_early_response(logits["early_response"], gt_early_response)
        prob_early_response = torch.softmax(logits["early_response"], dim=1)[:, 1]
        y_hat_response = torch.argmax(logits["early_response"], dim=1)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        
        self.log(f"{stage}_early_response_acc", self.acc(y_hat_response, gt_early_response), prog_bar=False)
        self.log(f"{stage}_early_response_auroc", self.auroc(prob_early_response, gt_early_response), prog_bar=True)
        self.log(f"{stage}_early_response_f1", self.f1(y_hat_response, gt_early_response), prog_bar=False)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # return optimizer
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (self.lr/self.max_epochs)*(self.max_epochs - epoch) if epoch < self.max_epochs else 0)
        return [optimizer], [scheduler]
    

##################################################################
##################### Siamese Network with Shared Weights #######
##################################################################

class CnnSiamese(pl.LightningModule):
    def __init__(self, config_dir: str, dropout=0.1, embed_dim=256):
        super(CnnSiamese, self).__init__()
        self.save_hyperparameters()
        config = read_yaml_file(config_dir)

        self.backbone = densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2)
        self.backbone.class_layers = torch.nn.Identity()
        self.pool = nn.AdaptiveAvgPool3d(1)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, config["resize"][0], config["resize"][1], config["resize"][2])  # small dummy volume
            feat = self.backbone.features(dummy)             # [1, C_feat, d, h, w]
            c_feat = feat.shape[1]

        self.proj = nn.Sequential(
            nn.Linear(c_feat, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
        )
        self.lr = config['lr']
        self.max_epochs = config['max_epochs']

        class_weights = torch.tensor([0.63, 1.37], dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)
        self.criterion_early_response = nn.CrossEntropyLoss(weight=self.class_weights)

        # Metrics
        self.acc = BinaryAccuracy()
        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)        # [B, C_feat, d, h, w]
        f = F.relu(f, inplace=True)
        f = self.pool(f).flatten(1)          # [B, C_feat]
        z = self.proj(f)                     # [B, embed_dim]
        return z
    
    def forward(self, batch):
        base_img = batch["base_img"].to(self.device)
        followup_img = batch["followup_img"].to(self.device)

        # pass through shared backbone
        z_base = self.encode(base_img)
        z_followup = self.encode(followup_img)
        
        fused = torch.cat([z_base, z_followup, z_followup - z_base, z_followup * z_base], dim=1)
        logits_early_response = self.head(fused).squeeze(1)
        logits = {"early_response": logits_early_response} #"survival": logits_overall_survival, 
        return logits

    def _shared_step(self, batch, stage):
        logits = self(batch)
        
        gt_early_response = batch["early_response"].long()
        loss = self.criterion_early_response(logits["early_response"], gt_early_response)
        prob_early_response = torch.softmax(logits["early_response"], dim=1)[:, 1]
        y_hat_response = torch.argmax(logits["early_response"], dim=1)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        
        self.log(f"{stage}_early_response_acc", self.acc(y_hat_response, gt_early_response), prog_bar=False)
        self.log(f"{stage}_early_response_auroc", self.auroc(prob_early_response, gt_early_response), prog_bar=True)
        self.log(f"{stage}_early_response_f1", self.f1(y_hat_response, gt_early_response), prog_bar=False)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # return optimizer
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (self.lr/self.max_epochs)*(self.max_epochs - epoch) if epoch < self.max_epochs else 0)
        return [optimizer], [scheduler]
    

if __name__ == "__main__":    # Example usage
    model = CnnSiamese(config_dir="../configs/cnn_config.yaml")
    batch = {
        "base_img": torch.randn(1, 1, 192, 192, 128),
        "followup_img": torch.randn(1, 1, 192, 192, 128),
        "early_response": torch.tensor([1]),
        "overall_survival_24m": torch.tensor([0])
    }
    output = model(batch)
    print(output)