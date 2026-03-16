import torch
from utils import compute_dice


def MR_test_model(model, test_loader):
        model.eval()
        output = {"y_true": [], "y_pred": [], "y_prob": [], 
                  "base": {"y_true": [], "y_pred": [], "y_prob": []}, 
                  "followup": {"y_true": [], "y_pred": [], "y_prob": []}}
        try:
            device = model.device
        except:
             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for batch in test_loader:
                base_logits, followup_logits = model(batch)
                logits = torch.cat([base_logits, followup_logits], dim=0)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                base_targets = batch['targets']['morph_score_base']
                followup_targets = batch['targets']['morph_score_followup']
                targets = torch.cat([base_targets, followup_targets], dim=0)

                output["y_pred"].extend(preds.cpu().numpy())
                output["y_prob"].extend(probs.cpu().numpy())
                output["y_true"].extend(targets.cpu().numpy())
                
                output["base"]["y_true"].extend(base_targets.cpu().numpy())
                output["base"]["y_pred"].extend(torch.argmax(base_logits, dim=1).cpu().numpy())
                output["base"]["y_prob"].extend(torch.softmax(base_logits, dim=1).cpu().numpy())
                
                output["followup"]["y_true"].extend(followup_targets.cpu().numpy())
                output["followup"]["y_pred"].extend(torch.argmax(followup_logits, dim=1).cpu().numpy())
                output["followup"]["y_prob"].extend(torch.softmax(followup_logits, dim=1).cpu().numpy())

        return output


def MR_rpn3d_test_model(model, test_loader):
        model.eval()
        output = {"y_true": [], "y_pred": [], "y_prob": [], 'base_dice': [], 'followup_dice': [],
                    "base": {"y_true": [], "y_pred": [], "y_prob": []},
                    "followup": {"y_true": [], "y_pred": [], "y_prob": []}
                  }
        with torch.no_grad():
            for batch in test_loader:
                base_logits, followup_logits, base_seg_logits, followup_seg_logits = model(batch)
                
                logits = torch.cat([base_logits, followup_logits], dim=0)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                base_targets = batch['targets']['morph_score_base']
                followup_targets = batch['targets']['morph_score_followup']
                targets = torch.cat([base_targets, followup_targets], dim=0)

                base_seg_gt = batch['base_seg']
                followup_seg_gt = batch['followup_seg']

                base_dice = compute_dice(base_seg_logits, base_seg_gt)
                followup_dice = compute_dice(followup_seg_logits, followup_seg_gt)

                output["y_pred"].extend(preds.cpu().numpy())
                output["y_prob"].extend(probs.cpu().numpy())
                output["y_true"].extend(targets.cpu().numpy())
                
                output["base"]["y_true"].extend(base_targets.cpu().numpy())
                output["base"]["y_pred"].extend(torch.argmax(base_logits, dim=1).cpu().numpy())
                output["base"]["y_prob"].extend(torch.softmax(base_logits, dim=1).cpu().numpy())
                
                output["followup"]["y_true"].extend(followup_targets.cpu().numpy())
                output["followup"]["y_pred"].extend(torch.argmax(followup_logits, dim=1).cpu().numpy())
                output["followup"]["y_prob"].extend(torch.softmax(followup_logits, dim=1).cpu().numpy())
                output["base_dice"].extend(base_dice)
                output["followup_dice"].extend(followup_dice)
        return output