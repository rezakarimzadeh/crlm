import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score

def read_yaml_file(file_path):
    import yaml
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
        
class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, M, L):
        super(AttentionMIL, self).__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = 1
        
        self.BN = MaskedBatchNorm1d(input_dim)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.M)
        )
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )


    def forward(self, x, pad_mask, attention=False):
        x = self.BN(x, pad_mask)
        # x: [B, T, F]
        H = self.feature_extractor(x)  # [B, T, H]
        A = self.attention(H).squeeze(-1)  # [B, T]
        # Apply mask before softmax
        A = A.masked_fill(pad_mask, float('-inf'))
        A = F.softmax(A, dim=1)  # [B, T]
        M = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # [B, H]
        M = M.view(M.size(0), -1)  # Flatten to [B, H]
        if attention:
            return M, A
        return M

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.classifier(x)

class RadiomicsMIL(pl.LightningModule):
    def __init__(self, features_dim: int, demographic_dim: int, config_dir: str):
        super(RadiomicsMIL, self).__init__()
        self.save_hyperparameters()
        dim = 128 #int((4/5)*features_dim)
        self.mil_model = AttentionMIL(features_dim, hidden_dim=dim, M=dim, L=dim)
        
        self.overall_survival_head = Classifier(input_dim=3*dim+demographic_dim, hidden_dim=dim//2, output_dim=2)
        self.early_response_head = Classifier(input_dim=3*dim+demographic_dim, hidden_dim=dim//2, output_dim=2)

        config = read_yaml_file(config_dir)
        self.lr = config['lr']
        self.max_epochs = config['max_epochs']  
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.acc = BinaryAccuracy()
        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()

    def forward(self, batch):
        base_emb = self.mil_model(x=batch["base"]["features"].to(self.device), pad_mask=batch["base"]["pad_mask"].to(self.device))
        followup_emb = self.mil_model(x=batch["followup"]["features"].to(self.device), pad_mask=batch["followup"]["pad_mask"].to(self.device))
        combined_emb = torch.cat([base_emb - followup_emb, base_emb, followup_emb, batch["demographic_info"].to(self.device)], dim=1)
        logits_overall_survival = self.overall_survival_head(combined_emb)
        logits_early_response = self.early_response_head(combined_emb)
        logits = {"survival": logits_overall_survival, "early_response": logits_early_response}
        return logits

    def _shared_step(self, batch, stage):
        logits = self(batch)
        gt_survival = batch["targets"]["overall_survival_24m"].long()
        gt_early_response = batch["targets"]["early_response"].long()
        loss = self.criterion(logits["survival"], gt_survival) + self.criterion(logits["early_response"], gt_early_response)
        y_hat_survival = torch.argmax(logits["survival"], dim=1)
        y_hat_response = torch.argmax(logits["early_response"], dim=1)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_overall_survival_acc", self.acc(y_hat_survival, gt_survival), prog_bar=False)
        self.log(f"{stage}_overall_survival_auroc", self.auroc(y_hat_survival, gt_survival), prog_bar=True)
        self.log(f"{stage}_overall_survival_f1", self.f1(y_hat_survival, gt_survival), prog_bar=False)
        
        self.log(f"{stage}_early_response_acc", self.acc(y_hat_response, gt_early_response), prog_bar=False)
        self.log(f"{stage}_early_response_auroc", self.auroc(y_hat_response, gt_early_response), prog_bar=True)
        self.log(f"{stage}_early_response_f1", self.f1(y_hat_response, gt_early_response), prog_bar=False)
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
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (self.lr/self.max_epochs)*(self.max_epochs - epoch) if epoch < self.max_epochs else 0)
        return [optimizer], [scheduler]


class MaskedBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x, pad_mask=None):
        """
        x: [B, T, F]  (must be float)
        pad_mask: [B, T] boolean, True for PAD, False for valid
        """
        x = x.float()  # BN needs float
        device = x.device
        B, T, F = x.shape

        assert F == self.bn.num_features, f"BN expects {self.bn.num_features} features, got {F}"

        if pad_mask is None:
            x_flat = x.reshape(B * T, F)
            x_bn = self.bn(x_flat)
            return x_bn.reshape(B, T, F)

        pad_mask = pad_mask.to(device=device, dtype=torch.bool)
        x_flat = x.reshape(B * T, F)

        mask_flat = pad_mask.reshape(B * T)     # True=pad
        valid_idx = ~mask_flat                  # True=valid

        n_valid = int(valid_idx.sum().item())
        if n_valid == 0:
            return x

        x_valid = x_flat[valid_idx]             # [N_valid, F]

        # Avoid BN instability with tiny batches in training
        if self.training and n_valid < 2:
            return x

        x_valid_bn = self.bn(x_valid)

        x_flat_out = x_flat.clone()
        x_flat_out[valid_idx] = x_valid_bn

        return x_flat_out.reshape(B, T, F)
