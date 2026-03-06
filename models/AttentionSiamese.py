import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score, MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x





class Encoder(nn.Module):

    def __init__(self, img_ch=1, base_features=32):
        super(Encoder, self).__init__()

        self.Conv1 = ConvBlock(img_ch, base_features)
        self.Conv2 = ConvBlock(base_features, base_features * 2)
        self.Conv3 = ConvBlock(base_features * 2, base_features * 4)
        self.Conv4 = ConvBlock(base_features * 4, base_features * 8)
        self.Conv5 = ConvBlock(base_features * 8, base_features * 16)

        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        return e1, e2, e3, e4, e5

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
           
        # self.fc = nn.Sequential(nn.Conv3d(in_channels, in_channels//2, 1, bias=False),
        #                        nn.ReLU(),
        #                        nn.Conv3d(in_channels//2, in_channels, 1, bias=False))
        self.fc = nn.Linear(in_channels, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, query):
        avg = self.avg_pool(query)
        max = self.max_pool(query)
        avg_out = self.fc(avg.flatten(1))  # [B, C]
        max_out = self.fc(max.flatten(1))  # [B, C]
        query_out = avg_out + max_out
        attention = self.sigmoid(query_out)
        return features * attention[:, :, None, None, None]  # [B, C, 1, 1, 1]


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, query):
        avg_out = torch.mean(query, dim=1, keepdim=True)
        max_out, _ = torch.max(query, dim=1, keepdim=True)
        query_out = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv1(query_out))
        return features * attention


class ChannelSpatialAttentionBlock(nn.Module):
    def __init__(self, dim):
        super(ChannelSpatialAttentionBlock, self).__init__()

        self.channel_attention = ChannelAttention(dim)
        self.spatial_attention = SpatialAttention()

    def forward(self, features, query):
        features = self.channel_attention(features, query)
        features = self.spatial_attention(features, query)
        return features

    

class SegDecoder(nn.Module):

    def __init__(self, base_features=32, output_ch=1):
        super(SegDecoder, self).__init__()

        self.base_features = base_features

        self.Up5 = UpConv(base_features * 16, base_features * 8)
        self.Att5 = ChannelSpatialAttentionBlock(dim=base_features * 8)
        self.UpConv5 = ConvBlock(base_features * 8, base_features * 8)

        self.Up4 = UpConv(base_features * 8, base_features * 4)
        self.Att4 = ChannelSpatialAttentionBlock(dim=base_features * 4)
        self.UpConv4 = ConvBlock(base_features * 4, base_features * 4)

        self.Up3 = UpConv(base_features * 4, base_features * 2)
        self.Att3 = ChannelSpatialAttentionBlock(dim=base_features * 2)
        self.UpConv3 = ConvBlock(base_features * 2, base_features * 2)

        self.Up2 = UpConv(base_features * 2, base_features)
        self.Att2 = ChannelSpatialAttentionBlock(dim=base_features)
        self.UpConv2 = ConvBlock(base_features, base_features)

        self.Conv = nn.Conv3d(base_features, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, e1, e2, e3, e4, e5):
        d5 = self.Up5(e5)

        s4 = self.Att5(features=e4, query=d5)
        d5_agg = s4 + d5 # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5_agg)

        d4 = self.Up4(d5)
        s3 = self.Att4(features=e3, query=d4)
        d4_agg = s3 + d4
        d4 = self.UpConv4(d4_agg)

        d3 = self.Up3(d4)
        s2 = self.Att3(features=e2, query=d3)
        d3_agg = s2 + d3
        d3 = self.UpConv3(d3_agg)

        d2 = self.Up2(d3)
        s1 = self.Att2(features=e1, query=d2)
        d2_agg = s1 + d2
        d2 = self.UpConv2(d2_agg)

        out = self.Conv(d2)
        return out 

class SegUNet(nn.Module):
    def __init__(self, img_ch=1, base_features=32, output_ch=1):
        super(SegUNet, self).__init__()

        self.encoder = Encoder(img_ch, base_features)
        self.decoder = SegDecoder(base_features, output_ch)
    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        out = self.decoder(e1, e2, e3, e4, e5)
        return out

class CrossChannelSpatialAttentionBlock(nn.Module):
    def __init__(self, dim):
        super(CrossChannelSpatialAttentionBlock, self).__init__()

        self.channel_attention_pre = ChannelAttention(dim)
        self.spatial_attention_pre = SpatialAttention()

        self.channel_attention_post = ChannelAttention(dim)
        self.spatial_attention_post = SpatialAttention()

        self.conv_block = ConvBlock(in_channels=3*dim, out_channels=dim)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)


    def forward(self, f1, f2):

        features_pre = self.channel_attention_pre(features=f1, query=f2)
        features_pre = self.spatial_attention_pre(features=features_pre, query=f2)

        features_post = self.channel_attention_post(features=f2, query=f1)
        features_post = self.spatial_attention_post(features=features_post, query=f1)

        aggregated = torch.cat([features_pre, features_post, features_pre-features_post], dim=1)
        aggregated = self.conv_block(aggregated)
        aggregated = self.avg_pool(aggregated).squeeze((2,3,4))  # [B, C]
        return aggregated  


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.BatchNorm1d(in_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//4),
            nn.BatchNorm1d(in_dim//4),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//4, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)
    

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
    

class CrossAttentionClassifier(nn.Module):
    def __init__(self, base_feature, demographic_dim, num_classes, deep_supervision):
        super(CrossAttentionClassifier, self).__init__()

        self.cross_attention_l1 = CrossChannelSpatialAttentionBlock(dim=base_feature)
        self.cross_attention_l2 = CrossChannelSpatialAttentionBlock(dim=base_feature*2)
        self.cross_attention_l3 = CrossChannelSpatialAttentionBlock(dim=base_feature*4)
        self.cross_attention_l4 = CrossChannelSpatialAttentionBlock(dim=base_feature*8)
        self.cross_attention_l5 = CrossChannelSpatialAttentionBlock(dim=base_feature*16)

        self.classifier = ClassifierHead(in_dim=31*base_feature+demographic_dim, num_classes=num_classes)
        self.projection_head = nn.Sequential(
            nn.BatchNorm1d(demographic_dim),
            nn.Linear(demographic_dim, demographic_dim),
            nn.ReLU(inplace=True)
        )

        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([SimpleClassifier(in_dim=base_feature*(2**i), num_classes=num_classes) for i in range(5)])       

    def forward(self, pre_features, post_features, demographic_info):
        pre_e1, pre_e2, pre_e3, pre_e4, pre_e5 = pre_features
        post_e1, post_e2, post_e3, post_e4, post_e5 = post_features
        attended_e1 = self.cross_attention_l1(pre_e1, post_e1)
        attended_e2 = self.cross_attention_l2(pre_e2, post_e2)
        attended_e3 = self.cross_attention_l3(pre_e3, post_e3)
        attended_e4 = self.cross_attention_l4(pre_e4, post_e4)
        attended_e5 = self.cross_attention_l5(pre_e5, post_e5)
        attended_features = torch.cat([attended_e1, attended_e2, attended_e3, attended_e4, attended_e5], dim=1)
        combined = torch.cat([attended_features, self.projection_head(demographic_info)], dim=1)
        logits = self.classifier(combined)
        
        deep_supervison_logits = None
        if self.deep_supervision:
            deep_supervison_logits = torch.mean(torch.stack([head(attended_e) for head, attended_e in zip(self.deep_supervision_heads, [attended_e1, attended_e2, attended_e3, attended_e4, attended_e5])]), dim=0)
        return logits, deep_supervison_logits


class AttentionSiameseMTLModel(nn.Module):
    def __init__(self, base_feature, demographic_dim, list_targets, list_num_classes, deep_supervision):
        super(AttentionSiameseMTLModel, self).__init__()

        self.encoder = Encoder(img_ch=1, base_features=base_feature)
        self.seg_decoder = SegDecoder(base_features=base_feature, output_ch=1)

        self.classifiers = nn.ModuleList()
        self.list_targets = list_targets

        for num_classes in list_num_classes:
            self.classifiers.append(CrossAttentionClassifier(base_feature=base_feature, demographic_dim=demographic_dim, num_classes=num_classes, deep_supervision=deep_supervision))
            

    def forward(self, pre_img, post_img, demographic_info):
        pre_features = self.encoder(pre_img)
        pre_seg_logits = self.seg_decoder(*pre_features)
    
        post_features = self.encoder(post_img)
        post_seg_logits = self.seg_decoder(*post_features)
        outputs = {}
        outputs['pre_seg_logits'] = pre_seg_logits
        outputs['post_seg_logits'] = post_seg_logits
        for target, classifier in zip(self.list_targets, self.classifiers):
            logits, deep_supervision_logits = classifier(pre_features, post_features, demographic_info)
            outputs[f'classifier_logits_{target}'] = logits
            if deep_supervision_logits is not None:
                outputs[f'deep_supervision_logits_{target}'] = deep_supervision_logits
        return outputs


def read_yaml_file(path: str) -> dict:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    

class AttentionMTLLightning(pl.LightningModule):
    def __init__(self, demographic_dim: int, config_dir: str):
        super(AttentionMTLLightning, self).__init__()
        self.save_hyperparameters()
        config = read_yaml_file(config_dir)
        self.lr = config['lr']
        self.max_epochs = config['max_epochs']

        self.attention_siamese = AttentionSiameseMTLModel(base_feature=config['base_features'], 
                                                          demographic_dim=demographic_dim, 
                                                          list_targets=config['list_targets'], 
                                                          list_num_classes=config['list_num_classes'], 
                                                          deep_supervision=config['deep_supervision'])

        self.target_keys = config['list_targets']
        self.deep_supervision = config['deep_supervision']
        print(f"Deep supervision enabled: {self.deep_supervision}")
        self.deep_supervision_weight = 0.1 

        

        class_weights = torch.tensor([0.63, 1.37], dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)
        self.criterion_early_recurrence = nn.CrossEntropyLoss(weight=self.class_weights) 

        self.criterion = nn.CrossEntropyLoss()
        self.seg_criterion = torch.nn.BCEWithLogitsLoss()
        # ---- Kendall MTL uncertainty weighting (all tasks are classification) ----
        all_tasks = config['list_targets'] + ["pre_segmentation", "post_segmentation"]
        self.task2idx = {k: i for i, k in enumerate(all_tasks)}
        # s_i = log(sigma_i^2) per task (learned)
        self.log_vars = nn.Parameter(torch.zeros(len(all_tasks)))  # initialized to log(1) = 0, so initial sigma^2 = 1

        # Metrics
        self.binary_acc = BinaryAccuracy()
        self.binary_auroc = BinaryAUROC()
        self.binary_f1 = BinaryF1Score()

        self.multi_acc = MulticlassAccuracy(num_classes=3)
        self.multi_auroc = MulticlassAUROC(num_classes=3)
        self.multi_f1 = MulticlassF1Score(num_classes=3, average="macro")

    def forward(self, batch):
        x_pre = batch["base_img"].to(self.device)        # [B,C,D,H,W]
        x_post = batch["followup_img"].to(self.device)   # [B,C,D,H,W]
        demographic_info = batch["demographic_info"].to(self.device)  # [B, demographic_dim]
        output_dict = self.attention_siamese(x_pre, x_post, demographic_info)
        return output_dict


    def _shared_step(self, batch, stage):
        output_dict = self(batch)

        classification_loss = 0.0

        for target_key in self.target_keys:
            gt_label = batch["targets"][target_key].long()

            # ===== BASE TASK LOSS (unchanged) =====
            if target_key == "early_recurrence":
                base_loss = self.criterion_early_recurrence(output_dict[f"classifier_logits_{target_key}"], gt_label)
                if self.deep_supervision:
                    base_loss = base_loss + self.deep_supervision_weight * self.criterion_early_recurrence(output_dict[f"deep_supervision_logits_{target_key}"], gt_label)

            elif target_key == "pathology":
                mask = gt_label != -1
                # if nothing valid, skip this task for this batch
                if mask.sum() == 0:
                    continue
                base_loss = self.criterion(output_dict[f"classifier_logits_{target_key}"][mask], gt_label[mask])
                if self.deep_supervision:
                    base_loss = base_loss + self.deep_supervision_weight * self.criterion(output_dict[f"deep_supervision_logits_{target_key}"][mask], gt_label[mask])
            else:
                base_loss = self.criterion(output_dict[f"classifier_logits_{target_key}"], gt_label)
                if self.deep_supervision:
                    base_loss = base_loss + self.deep_supervision_weight * self.criterion(output_dict[f"deep_supervision_logits_{target_key}"], gt_label)

            # ===== KENDALL WEIGHTING (classification form) =====
            i = self.task2idx[target_key]
            s = self.log_vars[i]                 # log(sigma^2)
            precision = torch.exp(-s)            # 1/sigma^2
            loss = precision * base_loss + 0.5 * s
            classification_loss = classification_loss + loss

            # (optional but useful) log sigma for monitoring
            self.log(f"{stage}_{target_key}_sigma", torch.exp(0.5 * s).detach(), prog_bar=False)

            # ===== METRICS (unchanged, but use base_loss for per-task loss logging) =====
            if target_key == "pathology":
                mask = gt_label != -1
                logits_valid = output_dict[f"classifier_logits_{target_key}"][mask]
                gt_valid = gt_label[mask]

                if len(gt_valid) > 0:
                    probs = torch.softmax(logits_valid, dim=1)
                    preds = torch.argmax(logits_valid, dim=1)

                    self.log(f"{stage}_{target_key}_acc", self.multi_acc(preds, gt_valid), prog_bar=False)
                    self.log(f"{stage}_{target_key}_auroc", self.multi_auroc(probs, gt_valid), prog_bar=False)
                    self.log(f"{stage}_{target_key}_f1", self.multi_f1(preds, gt_valid))

            else:
                probs = torch.softmax(output_dict[f"classifier_logits_{target_key}"], dim=1)[:, 1]
                preds = torch.argmax(output_dict[f"classifier_logits_{target_key}"], dim=1)

                self.log(f"{stage}_{target_key}_acc", self.binary_acc(preds, gt_label))
                self.log(f"{stage}_{target_key}_auroc", self.binary_auroc(probs, gt_label), prog_bar=False)
                self.log(f"{stage}_{target_key}_f1", self.binary_f1(preds, gt_label))

            self.log(f"{stage}_{target_key}_loss", loss, prog_bar=True)
        
        pre_seg_loss = self.seg_criterion(output_dict["pre_seg_logits"], batch["base_seg"].to(self.device))
        post_seg_loss = self.seg_criterion(output_dict["post_seg_logits"], batch["followup_seg"].to(self.device))
        
        i = self.task2idx["pre_segmentation"]
        s_pre = self.log_vars[i]
        precision_pre = torch.exp(-s_pre)
        seg_pre_loss_weighted = precision_pre * pre_seg_loss + 0.5 * s_pre
        self.log(f"{stage}_pre_segmentation_sigma", torch.exp(0.5 * s_pre).detach(), prog_bar=False)
        
        i = self.task2idx["post_segmentation"]
        s_post = self.log_vars[i]
        precision_post = torch.exp(-s_post)
        seg_post_loss_weighted = precision_post * post_seg_loss + 0.5 * s_post
        self.log(f"{stage}_post_segmentation_sigma", torch.exp(0.5 * s_post).detach(), prog_bar=False)
        
        seg_loss = seg_pre_loss_weighted + seg_post_loss_weighted
        self.log(f"{stage}_seg_loss", seg_loss, prog_bar=True)

        total_loss = classification_loss + seg_loss
        self.log(f"{stage}_loss", total_loss, prog_bar=True)
        return total_loss
    
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
     
if __name__ == "__main__":
    model = SegUNet(img_ch=1, base_features=32, output_ch=1)
    x = torch.randn(2, 1, 64, 64, 64)  # Example input tensor
    output = model(x)
    # number of trainable parameters
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {parameters}")
    print(output.shape)  # Should be [2, 1, 64, 64, 64]

class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv3d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):

    def __init__(self, img_ch=1, base_features=32, output_ch=1):
        super(AttentionUNet, self).__init__()
        self.base_features = base_features
        self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)

        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        return out