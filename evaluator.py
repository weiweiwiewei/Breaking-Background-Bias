"""Evaluation utilities for classification, OOD detection, and open-set recognition."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_fuse_prototypes(model, train_loader, num_classes, device):
    """Compute fused feature prototypes μ_c^fuse using projected features (feat_proj)."""
    model.eval()
    feat_dim = 384
    mu_sum = torch.zeros(num_classes, feat_dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for foreground, background, labels in tqdm(train_loader, desc='Fuse prototypes', ncols=100):
            foreground = foreground.to(device)
            background = background.to(device)
            labels = labels.to(device)
            
           
            _, fused_feat, _ = model(
                foreground, background, multi_instance=False, return_features=True
            )  
            for i in range(labels.size(0)):
                c = labels[i].item()
                mu_sum[c] += fused_feat[i]
                counts[c] += 1
    
    counts = counts.clamp(min=1.0).unsqueeze(1)
    mu_fuse = mu_sum / counts  # [C, 384]
    return mu_fuse


def get_ood_loader(ood_loader, ood_batch_size=None):
    """Return an OOD DataLoader with the desired batch size if needed."""
    if ood_batch_size is not None and ood_batch_size > ood_loader.batch_size:
        return DataLoader(
            ood_loader.dataset, 
            batch_size=ood_batch_size,
            shuffle=False,
            num_workers=ood_loader.num_workers,
            pin_memory=ood_loader.pin_memory
        )
    else:
        return ood_loader


def compute_fd_fuse(fused_feat, mu_fuse):
    """Compute FD_fuse distances to nearest fused prototype."""
    diff = fused_feat.unsqueeze(1) - mu_fuse.unsqueeze(0)  # [B, C, D]
    dists = diff.norm(dim=-1)  # [B, C]
    FD, min_idx = dists.min(dim=1)  # [B]
    
    return FD, min_idx


def evaluate(model, test_loader, device):
    """Evaluate model using the classifier head."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for foreground, background, labels in tqdm(test_loader, desc='Evaluating (Classifier)', ncols=100):
            foreground = foreground.to(device)
            background = background.to(device)
            labels = labels.to(device)
            
            logits = model(foreground, background)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_probs, all_labels


def evaluate_with_prototype(model, test_loader, train_loader, num_classes, device, 
                           mu_fuse=None):
    """Evaluate model using prototype distances instead of classifier logits."""
    model.eval()
    
    
    if mu_fuse is None:
        mu_fuse = compute_fuse_prototypes(model, train_loader, num_classes, device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for foreground, background, labels in tqdm(test_loader, desc='Evaluating (Prototype)', ncols=100):
            foreground = foreground.to(device)
            background = background.to(device)
            labels = labels.to(device)
            
            
            _, fused_feat, _ = model(
                foreground, background, multi_instance=False, return_features=True
            )
            
            
            _, min_idx = compute_fd_fuse(fused_feat, mu_fuse)
            
            all_preds.extend(min_idx.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
         
            diff = fused_feat.unsqueeze(1) - mu_fuse.unsqueeze(0)  # [B, C, D]
            dists = diff.norm(dim=-1)  # [B, C]
            
            logits = -dists  
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_probs, all_labels


def compute_oscr(pred, scores, labels):
    """Compute OSCR (Open Set Classification Rate) from predictions, scores, and labels."""
    unk_mask = labels < 0
    id_mask = labels >= 0
    
    if unk_mask.sum() == 0 or id_mask.sum() == 0:
        return 0.0
    
    unk_scores = scores[unk_mask]
    id_scores = scores[id_mask]
    id_pred = pred[id_mask]
    id_labels = labels[id_mask]
    id_correct = (id_pred == id_labels)
    
    unk_ct = unk_mask.sum()
    id_ct = id_mask.sum()
    
    def fpr(thr):
        
        return (unk_scores > thr).sum() / unk_ct if unk_ct > 0 else 0
    
    def ccr(thr):
        
        ac_cond = (id_scores > thr) & id_correct
        return ac_cond.sum() / id_ct if id_ct > 0 else 0
    
    sorted_scores = -np.sort(-scores)
    
    CCR = [0]
    FPR = [0]
    
    for s in sorted_scores:
        CCR.append(ccr(s))
        FPR.append(fpr(s))
    
    CCR += [1]
    FPR += [1]
    
    
    ROC = sorted(zip(FPR, CCR), reverse=True)
    OSCR = 0
    for j in range(len(CCR) - 1):
        h = ROC[j][0] - ROC[j+1][0]
        w = (ROC[j][1] + ROC[j+1][1]) / 2.0
        OSCR += h * w
    
    return OSCR


def evaluate_open_set_recognition(model, test_loader, ood_loader, train_loader, 
                                  num_classes, device, percentile=95, 
                                  ood_batch_size=None, mu_fuse=None):
   
    model.eval()
    
    
    if mu_fuse is None:
        mu_fuse = compute_fuse_prototypes(model, train_loader, num_classes, device)
    
    
    all_fused_feats = []
    all_labels = []
    all_raw_preds = [] 
    
  
    with torch.no_grad():
        for foreground, background, labels in tqdm(test_loader, desc='ID Features (OSR)', ncols=100):
            foreground = foreground.to(device)
            background = background.to(device)
            labels = labels.to(device)
            
           
            logits, fused_feat, _ = model(
                foreground, background, multi_instance=False, return_features=True
            )
            _, raw_pred = torch.max(logits, 1)
            
            all_fused_feats.append(fused_feat.cpu())
            all_labels.extend(labels.cpu().numpy())
            all_raw_preds.extend(raw_pred.cpu().numpy())
    
 
    ood_loader_to_use = get_ood_loader(ood_loader, ood_batch_size)
    
    with torch.no_grad():
        for foreground, background, _ in tqdm(ood_loader_to_use, desc='OOD Features (OSR)', ncols=100):
            foreground = foreground.to(device)
            background = background.to(device)
           
            logits, fused_feat, _ = model(
                foreground, background, multi_instance=False, return_features=True
            )
            _, raw_pred = torch.max(logits, 1)
            batch_size = raw_pred.size(0)
            
            all_fused_feats.append(fused_feat.cpu())
            all_labels.extend([-1] * batch_size) 
            all_raw_preds.extend(raw_pred.cpu().numpy())
    
    
    all_fused_feats = torch.cat(all_fused_feats, dim=0).to(device)
    all_labels = np.array(all_labels)
    all_raw_preds = np.array(all_raw_preds)
    
    
    FD, min_idx = compute_fd_fuse(all_fused_feats, mu_fuse)
    FD = FD.cpu().numpy()
    min_idx = min_idx.cpu().numpy()
    
    
    ood_scores = -FD  
    
    
    oscr = compute_oscr(all_raw_preds, ood_scores, all_labels)
    
    return {
        'oscr': oscr
    }


def evaluate_ood(model, test_loader, ood_loader, train_loader, num_classes, device, 
                 delta_f=None, ood_batch_size=None, use_fuse_features=True,
                 percentile=95, mu_fuse=None):
    
    model.eval()
    
    if use_fuse_features:
        if mu_fuse is None:
            mu_fuse = compute_fuse_prototypes(model, train_loader, num_classes, device)
        
        id_fused_feats = []
        with torch.no_grad():
            for foreground, background, _ in tqdm(test_loader, desc='ID Features', ncols=100):
                foreground = foreground.to(device)
                background = background.to(device)
                _, fused_feat, _ = model(
                    foreground, background, multi_instance=False, return_features=True
                )
                id_fused_feats.append(fused_feat.cpu())
        
        id_fused_feats = torch.cat(id_fused_feats, dim=0).to(device)  # [N_id, 384]
        
        id_FD, _ = compute_fd_fuse(id_fused_feats, mu_fuse)
        id_FD = id_FD.cpu().numpy()
        
        ood_fused_feats = []
        ood_loader_to_use = get_ood_loader(ood_loader, ood_batch_size)
        
        if len(ood_loader_to_use.dataset) == 0:
            raise ValueError(f"OOD dataset is empty. Check path: {ood_loader.dataset.imagenet_root}, split: {ood_loader.dataset.split}")

        with torch.no_grad():
            for foreground, background, _ in tqdm(ood_loader_to_use, desc='OOD Features', ncols=100):
                foreground = foreground.to(device)
                background = background.to(device)
                _, fused_feat, _ = model(
                    foreground, background, multi_instance=False, return_features=True
                )
                ood_fused_feats.append(fused_feat.cpu())
        
        if len(ood_fused_feats) == 0:
            raise ValueError(f"No OOD features collected. Dataset size: {len(ood_loader_to_use.dataset)}.")
        
        ood_fused_feats = torch.cat(ood_fused_feats, dim=0).to(device)  # [N_ood, 384]
        
        ood_FD, _ = compute_fd_fuse(ood_fused_feats, mu_fuse)
        ood_FD = ood_FD.cpu().numpy()
        
        id_scores = -id_FD
        ood_scores = -ood_FD

        if delta_f is None:
           
            delta_f = np.percentile(id_FD, percentile)
        
    else:
       
        id_scores = []
        with torch.no_grad():
            for foreground, background, _ in tqdm(test_loader, desc='ID Scores', ncols=100):
                foreground = foreground.to(device)
                background = background.to(device)
                logits = model(foreground, background)
                probs = torch.softmax(logits, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                id_scores.extend(max_probs.cpu().numpy())
        
        ood_scores = []
        if ood_batch_size is not None and ood_batch_size > ood_loader.batch_size:
            ood_loader_large = DataLoader(
                ood_loader.dataset, 
                batch_size=ood_batch_size,
                shuffle=False,
                num_workers=ood_loader.num_workers,
                pin_memory=ood_loader.pin_memory
            )
            ood_loader_to_use = ood_loader_large
        else:
            ood_loader_to_use = ood_loader
        
        if len(ood_loader_to_use.dataset) == 0:
            raise ValueError(f"OOD dataset is empty. Check path: {ood_loader.dataset.imagenet_root}, split: {ood_loader.dataset.split}")

        with torch.no_grad():
            for foreground, background, _ in tqdm(ood_loader_to_use, desc='OOD Scores', ncols=100):
                foreground = foreground.to(device)
                background = background.to(device)
                logits = model(foreground, background)
                probs = torch.softmax(logits, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                ood_scores.extend(max_probs.cpu().numpy())
        
       
        if len(ood_scores) == 0:
            raise ValueError(f"No OOD scores collected. Dataset size: {len(ood_loader_to_use.dataset)}.")
        
        id_scores = np.array(id_scores)
        ood_scores = np.array(ood_scores)
        delta_f = None
    
  
    id_labels = np.ones(len(id_scores))
    ood_labels = np.zeros(len(ood_scores))
    
    all_scores = np.concatenate([id_scores, ood_scores])
    all_labels = np.concatenate([id_labels, ood_labels])
    
  
    auroc = roc_auc_score(all_labels, all_scores)
    
    return auroc, delta_f

n(id_scores))
    ood_labels = np.zeros(len(ood_scores))
    
    all_scores = np.concatenate([id_scores, ood_scores])
    all_labels = np.concatenate([id_labels, ood_labels])
    
  
    auroc = roc_auc_score(all_labels, all_scores)
    
    return auroc, delta_f

