import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAUROC, BinaryF1Score,
    MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
)
from torch.optim.lr_scheduler import CosineAnnealingLR


def read_yaml_file(file_path):
    import yaml
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, outputdim):
        super(FeatureExtractor, self).__init__()

        self.extractor1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )
        self.bn1 = MaskedBatchNorm1d(hidden_dim)
        self.extractor2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, outputdim)
        )

    def forward(self, x, mask):
        x = self.extractor1(x)
        x = self.bn1(x, mask)
        x = self.extractor2(x)
        return x
        
     

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, M, L):
        super(AttentionMIL, self).__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = 1
        
        self.BN = MaskedBatchNorm1d(input_dim)
        # self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, self.M)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(0.1),
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
        # H = self.feature_extractor(x, pad_mask)
        A = self.attention(H).squeeze(-1)  # [B, T]
        # Apply mask before softmax
        A = A.masked_fill(pad_mask, float('-inf'))
        A = F.softmax(A, dim=1)  # [B, T]
        M = torch.bmm(A.unsqueeze(1), H).squeeze(1)  # [B, H]
        M = M.view(M.size(0), -1)  # Flatten to [B, H]
        if attention:
            return M, A
        return M


class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, M, L, dropout=0.2):
        super().__init__()
        self.M = M
        self.BN = MaskedBatchNorm1d(input_dim)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, M),
            # nn.LayerNorm(M),
            # nn.ReLU(),
            # nn.Dropout(dropout),
        )

        self.attention_V = nn.Linear(M, L)
        self.attention_U = nn.Linear(M, L)
        self.attention_w = nn.Linear(L, 1)

    def forward(self, x, pad_mask, attention=False):
        x = self.BN(x, pad_mask)
        H = self.feature_extractor(x)                      # [B,T,M]

        A_V = torch.tanh(self.attention_V(H))
        A_U = torch.sigmoid(self.attention_U(H))
        A = self.attention_w(A_V * A_U).squeeze(-1)       # [B,T]

        A = A.masked_fill(pad_mask, -1e9)
        A = torch.softmax(A, dim=1)

        M = torch.bmm(A.unsqueeze(1), H).squeeze(1)       # [B,M]

        if attention:
            return M, A
        return M
    

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            # nn.ReLU(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x):
        return self.classifier(x)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProjectionHead, self).__init__()

        self.projector = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.projector(x)
    
class RadiomicsMIL(pl.LightningModule):
    def __init__(self, features_dim: int, demographic_dim: int, config_dir: str, target_key: str):
        super(RadiomicsMIL, self).__init__()
        self.save_hyperparameters()

        self.target_key = target_key
        self.multi_class = True if self.target_key == "morph_response" else False
        if self.multi_class:
            output_dim = 3
        else:
            output_dim = 2

        dim = features_dim*2 #int((4/5)*features_dim)
        # self.mil_model = AttentionMIL(features_dim, hidden_dim=dim, M=dim, L=dim)
        self.mil_model = GatedAttentionMIL(features_dim, hidden_dim=dim, M=dim, L=dim)
        
        self.classifier_head = Classifier(input_dim=2*dim+demographic_dim, hidden_dim=dim, output_dim=output_dim)
        self.projection_head_demographic = ProjectionHead(input_dim=demographic_dim, hidden_dim=demographic_dim, output_dim=demographic_dim)

        self.ds_head_nodes = nn.Linear(2*dim, output_dim)
        self.ds_head_demographic = nn.Linear(demographic_dim, output_dim)


        config = read_yaml_file(config_dir)
        self.lr = config['lr']
        self.max_epochs = config['max_epochs']

        if self.target_key == "early_recurrence":
            class_weights = torch.tensor([0.63, 1.37], dtype=torch.float32)
            self.register_buffer("class_weights", class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights) 
        elif self.target_key == "morph_response":
            class_weights = torch.tensor([0.55, 1.42, 2.10], dtype=torch.float32)
            self.register_buffer("class_weights", class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Metrics
        if self.multi_class:
            self.acc = MulticlassAccuracy(num_classes=output_dim)
            self.auroc = MulticlassAUROC(num_classes=output_dim)
            self.f1 = MulticlassF1Score(num_classes=output_dim)
        else:
            self.acc = BinaryAccuracy()
            self.auroc = BinaryAUROC()
            self.f1 = BinaryF1Score()

    def forward(self, batch, deep_supervision=False):
        base_emb = self.mil_model(x=batch["base"]["features"].to(self.device), pad_mask=batch["base"]["pad_mask"].to(self.device))
        followup_emb = self.mil_model(x=batch["followup"]["features"].to(self.device), pad_mask=batch["followup"]["pad_mask"].to(self.device))
        demographic_emb = self.projection_head_demographic(batch["demographic_info"].to(self.device))
        combined_emb = torch.cat([base_emb, followup_emb, demographic_emb], dim=1)
        logits = self.classifier_head(combined_emb)
        if deep_supervision:
            return logits, self.ds_head_nodes(torch.cat([base_emb, followup_emb], dim=1)), self.ds_head_demographic(demographic_emb)
        else:
            return logits

    def _shared_step(self, batch, stage):
        logits, nodes_ds, demographic_ds = self(batch, deep_supervision=True)
        gt_label = batch["targets"][self.target_key].long()
        mask = gt_label != -1
        logits = logits[mask]
        gt_label = gt_label[mask]
        nodes_ds = nodes_ds[mask]
        demographic_ds = demographic_ds[mask]
        if gt_label.numel() == 0:
            return None
        
        ds_loss = self.criterion(nodes_ds, gt_label) + self.criterion(demographic_ds, gt_label)

        main_loss = self.criterion(logits, gt_label)
        if stage == "train":
            loss = main_loss + 0.2*ds_loss
        else:
            loss = main_loss
        y_hat = torch.argmax(logits, dim=1)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc", self.acc(y_hat, gt_label), prog_bar=False)
        self.log(f"{stage}_f1", self.f1(y_hat, gt_label), prog_bar=False)
        if self.multi_class:
            probs = torch.softmax(logits, dim=1)
            self.log(f"{stage}_auroc", self.auroc(probs, gt_label), prog_bar=True)
        else:
            probs = torch.softmax(logits, dim=1)[:, 1]
            self.log(f"{stage}_auroc", self.auroc(probs, gt_label), prog_bar=True)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # return optimizer
        scheduler = LambdaLR(
                    optimizer,
                    lr_lambda=lambda epoch: max(0.0, (self.max_epochs - epoch) / self.max_epochs)
                )
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


class MorphScoreRadiomicsMIL(pl.LightningModule):
    def __init__(self, features_dim: int, demographic_dim: int, config_dir: str):
        super(MorphScoreRadiomicsMIL, self).__init__()
        self.save_hyperparameters()

        output_dim = 3

        dim = features_dim*2 #int((4/5)*features_dim)
        # self.mil_model = AttentionMIL(features_dim, hidden_dim=dim, M=dim, L=dim)
        self.mil_model = GatedAttentionMIL(features_dim, hidden_dim=dim, M=dim, L=dim)
        
        self.classifier_head = Classifier(input_dim=dim+demographic_dim, hidden_dim=dim, output_dim=output_dim)
        self.projection_head_demographic = ProjectionHead(input_dim=demographic_dim, hidden_dim=demographic_dim, output_dim=demographic_dim)


        config = read_yaml_file(config_dir)
        self.lr = config['lr']
        self.max_epochs = config['max_epochs']

        class_weights = torch.tensor([1.15, 1.93, 0.62], dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        self.acc = MulticlassAccuracy(num_classes=output_dim)
        self.auroc = MulticlassAUROC(num_classes=output_dim)
        self.f1 = MulticlassF1Score(num_classes=output_dim)


    def forward(self, batch):
        base_emb = self.mil_model(x=batch["base"]["features"].to(self.device), pad_mask=batch["base"]["pad_mask"].to(self.device))
        followup_emb = self.mil_model(x=batch["followup"]["features"].to(self.device), pad_mask=batch["followup"]["pad_mask"].to(self.device))
        demographic_emb = self.projection_head_demographic(batch["demographic_info"].to(self.device))

        base_combined_emb = torch.cat([base_emb, demographic_emb], dim=1)
        followup_combined_emb = torch.cat([followup_emb, demographic_emb], dim=1)
        base_logits = self.classifier_head(base_combined_emb)
        followup_logits = self.classifier_head(followup_combined_emb)

        return base_logits, followup_logits

    def _shared_step(self, batch, stage):
        base_logits, followup_logits = self(batch)
        logits = torch.cat([base_logits, followup_logits], dim=0)
        gt_label = torch.cat([batch["targets"]["morph_score_base"].long(), batch["targets"]["morph_score_followup"].long()], dim=0)

        # masking the outputs and labels where gt_label is -1 (indicating missing label) so they do not contribute to loss or metrics
        mask = gt_label != -1
        logits = logits[mask]
        gt_label = gt_label[mask]
        
        if gt_label.numel() == 0:
            return None
        
        loss = self.criterion(logits, gt_label)
        y_hat = torch.argmax(logits, dim=1)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_acc", self.acc(y_hat, gt_label), prog_bar=False)
        self.log(f"{stage}_f1", self.f1(y_hat, gt_label), prog_bar=False)
        probs = torch.softmax(logits, dim=1)
        self.log(f"{stage}_auroc", self.auroc(probs, gt_label), prog_bar=True)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        # return optimizer
        scheduler = LambdaLR(
                    optimizer,
                    lr_lambda=lambda epoch: max(0.0, (self.max_epochs - epoch) / self.max_epochs)
                )
        # scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        
        return [optimizer], [scheduler]