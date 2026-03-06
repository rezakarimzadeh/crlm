
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
import torchvision

def read_yaml_file(file_path):
    import yaml
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)



class GoogLeNetFeature(nn.Module):
    """1024-d feature vector from GoogLeNet (Inception v1) pretrained on ImageNet."""
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None

        # IMPORTANT: with pretrained weights, aux_logits must be True
        m = torchvision.models.googlenet(weights=weights, aux_logits=True)

        # remove classifier (main head)
        m.fc = nn.Identity()

        self.model = m

    def forward(self, x):  # x: [B,3,224,224]
        out = self.model(x)

        # When training AND aux_logits=True, torchvision returns GoogLeNetOutputs(logits, aux2, aux1)
        # When eval, it returns just logits.
        if isinstance(out, tuple) or hasattr(out, "logits"):
            out = out.logits  # [B,1024]

        return out


class GooglenetLSTM(pl.LightningModule):
    def __init__(self, config_dir: str, target_key: str):
        super(GooglenetLSTM, self).__init__()
        self.save_hyperparameters()
        self.cnn = GoogLeNetFeature(pretrained=True)

        self.rnn = nn.LSTM(
            input_size=1024,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * 256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        self.target_key = target_key

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

    # -------- diameter-weighted aggregation (per patient) --------
    def agg(self, emb, batch_idxes, diameters, B):
        # emb: [N, D], batch_idxes: [N], diameters: [N]
        batch_idxes = torch.as_tensor(batch_idxes, device=emb.device, dtype=torch.long)
        diameters = torch.as_tensor(diameters, device=emb.device, dtype=emb.dtype)

        # weighted sum: sum_i (d_i * e_i) for each patient
        weighted = emb * diameters.unsqueeze(1)  # [N, D]
        out = torch.zeros(B, emb.size(1), device=emb.device, dtype=emb.dtype)
        out.index_add_(0, batch_idxes, weighted)

        # normalize by sum of diameters per patient
        denom = torch.zeros(B, device=emb.device, dtype=emb.dtype)
        denom.index_add_(0, batch_idxes, diameters)
        out = out / denom.clamp_min(1e-6).unsqueeze(1)
        return out  # [B, D]

    def forward(self, batch):
        
        # Encode each lesion/crop -> (N_base, 1024) and (N_followup, 1024)
        base_emb = self.cnn(batch["base"]["img"].to(self.device))         # [N_base, 1024]
        followup_emb = self.cnn(batch["followup"]["img"].to(self.device)) # [N_followup, 1024]

        B = len(batch["patient_ids"])  # number of patients in this batch

        base_emb = self.agg(base_emb, batch["base"]["batch_idxes"].to(self.device), batch["base"]["diameters"].to(self.device), B)            # [B, 1024]
        followup_emb = self.agg(followup_emb, batch["followup"]["batch_idxes"].to(self.device), batch["followup"]["diameters"].to(self.device), B)  # [B, 1024]

        # LSTM wants (B, T, D). Here T=2: [base, followup]
        base_emb = base_emb.unsqueeze(1)         # [B, 1, 1024]
        followup_emb = followup_emb.unsqueeze(1) # [B, 1, 1024]
        combined_emb = torch.cat([base_emb, followup_emb], dim=1)  # [B, 2, 1024]

        lstm_out, _ = self.rnn(combined_emb)
        combined_emb = lstm_out[:, -1, :]  # [B, 2*256] because bidirectional
        logits = self.head(combined_emb)
        return logits

    def _shared_step(self, batch, stage):
        logits = self(batch)
        gt_label = batch["targets"][self.target_key].long()

        loss = self.criterion(logits, gt_label)
        prob_pos = torch.softmax(logits, dim=1)[:, 1]
        y_hat = torch.argmax(logits, dim=1)

        bs = len(batch["patient_ids"])  # patient-level batch size

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}_acc", self.acc(y_hat, gt_label), on_epoch=True, batch_size=bs)
        self.log(f"{stage}_auroc", self.auroc(prob_pos, gt_label), prog_bar=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}_f1", self.f1(y_hat, gt_label), on_epoch=True, batch_size=bs)
        return loss
    
    def get_attentions(self, batch):
        _, base_attention = self.mil_model(x=batch["base"]["features"].to(self.device), pad_mask=batch["base"]["pad_mask"].to(self.device), attention=True)
        _, followup_attention = self.mil_model(x=batch["followup"]["features"].to(self.device), pad_mask=batch["followup"]["pad_mask"].to(self.device), attention=True)
        return base_attention, followup_attention
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # return optimizer
        scheduler = LambdaLR(
                    optimizer,
                    lr_lambda=lambda epoch: max(0.0, (self.max_epochs - epoch) / self.max_epochs)
                )
        return [optimizer], [scheduler]