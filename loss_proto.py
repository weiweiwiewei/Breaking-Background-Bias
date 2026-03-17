"""Prototype-based three-level ranking loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RankLossProto(nn.Module):
    
    def __init__(self, m1=0.1, m2=0.2, beta=1.0, tau=0.1, K=16, K_prime=16, 
                 use_similarity=True, scale=20.0):
        super(RankLossProto, self).__init__()
        self.m1 = m1
        self.m2 = m2
        self.beta = beta
        self.tau = tau
        self.K = K
        self.K_prime = K_prime
        self.use_similarity = use_similarity
        self.scale = scale
    
    def compute_prototype_score(self, fused_feat, prototype, target_labels):
        if self.use_similarity:
            feat_norm = F.normalize(fused_feat, dim=-1)  # [B, D] 或 [B, N, D]
            proto_norm = F.normalize(prototype, dim=1)  # [C, D]
            # score(x, c) = scale * cos(f(x), w_c)
            if fused_feat.dim() == 2:
                # [B, D] @ [C, D].t() -> [B, C]
                cosine = feat_norm @ proto_norm.t()  # [B, C]
                scores = self.scale * cosine.gather(1, target_labels.unsqueeze(1)).squeeze(1)  # [B]
            else:
                # [B, N, D] -> [B*N, D]
                B, N, D = fused_feat.shape
                feat_flat = feat_norm.view(-1, D)  # [B*N, D]
                cosine_flat = feat_flat @ proto_norm.t()  # [B*N, C]
                target_flat = target_labels.view(-1)  # [B*N]
                scores_flat = self.scale * cosine_flat.gather(1, target_flat.unsqueeze(1)).squeeze(1)  # [B*N]
                scores = scores_flat.view(B, N)  # [B, N]
        else:
            # d(x, c) = ||f(x) - w_c||_2
            if fused_feat.dim() == 2:
                target_proto = prototype[target_labels]  # [B, D]
                diff = fused_feat - target_proto
                scores = diff.norm(dim=1)  # [B]
            else:
                # [B, N, D] -> [B*N, D]
                B, N, D = fused_feat.shape
                feat_flat = fused_feat.view(-1, D)  # [B*N, D]
                target_flat = target_labels.view(-1)  # [B*N]
                target_proto = prototype[target_flat]  # [B*N, D]
                diff = feat_flat - target_proto
                scores_flat = diff.norm(dim=1)  # [B*N]
                scores = scores_flat.view(B, N)  # [B, N]
        
        return scores
    
    def forward(self, fused_feat_pp, fused_feat_pm, fused_feat_mp, prototype, labels):
        B = labels.size(0)
        N_pm = fused_feat_pm.size(1) if fused_feat_pm.size(1) > 0 else 0
        N_mp = fused_feat_mp.size(1) if fused_feat_mp.size(1) > 0 else 0
        
        score_pp = self.compute_prototype_score(fused_feat_pp, prototype, labels)  # [B]
        
        use_pm = (self.K > 0) and (N_pm > 0)
        use_mp = (self.K_prime > 0) and (N_mp > 0)
        
        if not use_pm and not use_mp:
            raise ValueError(f"K={self.K} and K'={self.K_prime} cannot both be 0. At least one must be > 0.")
        
        if use_pm:
            labels_pm = labels.unsqueeze(1).expand(B, N_pm).contiguous()  # [B, N_pm]
            score_pm_all = self.compute_prototype_score(fused_feat_pm, prototype, labels_pm)  # [B, N_pm]
            k_pm = min(self.K, N_pm)
            if self.use_similarity:
                score_pm_topk, _ = torch.topk(score_pm_all, k=k_pm, dim=1)  # [B, k_pm]
                score_pm = score_pm_topk.mean(dim=1)  # [B]
            else:
                score_pm_topk, _ = torch.topk(score_pm_all, k=k_pm, dim=1, largest=False)  # [B, k_pm]
                score_pm = score_pm_topk.mean(dim=1)  # [B]
        else:
            score_pm = None
        
        if use_mp:
            labels_mp = labels.unsqueeze(1).expand(B, N_mp).contiguous()  # [B, N_mp]
            score_mp_all = self.compute_prototype_score(fused_feat_mp, prototype, labels_mp)  # [B, N_mp]
            k_mp = min(self.K_prime, N_mp)
            if self.use_similarity:
                score_mp_topk, _ = torch.topk(score_mp_all, k=k_mp, dim=1)  # [B, k_mp]
                score_mp = score_mp_topk.mean(dim=1)  # [B]
            else:
                score_mp_topk, _ = torch.topk(score_mp_all, k=k_mp, dim=1, largest=False)  # [B, k_mp]
                score_mp = score_mp_topk.mean(dim=1)  # [B]
        else:
            score_mp = None
        
        if self.use_similarity:
            if use_pm and use_mp:
                rank_loss_1 = F.softplus((self.m1 + score_pm - score_pp) / self.tau)  # [B]
                rank_loss_2 = self.beta * F.softplus((self.m2 + score_mp - score_pm) / self.tau)  # [B]
            elif use_pm:
                rank_loss_1 = F.softplus((self.m1 + score_pm - score_pp) / self.tau)  # [B]
                rank_loss_2 = torch.zeros_like(rank_loss_1)  # [B]
            elif use_mp:
                rank_loss_1 = torch.zeros(B, device=score_pp.device)  # [B]
                rank_loss_2 = F.softplus((self.m2 + score_mp - score_pp) / self.tau)  # [B]
        else:
            if use_pm and use_mp:
                rank_loss_1 = F.softplus((self.m1 + score_pp - score_pm) / self.tau)  # [B]
                rank_loss_2 = self.beta * F.softplus((self.m2 + score_pm - score_mp) / self.tau)  # [B]
            elif use_pm:
                rank_loss_1 = F.softplus((self.m1 + score_pp - score_pm) / self.tau)  # [B]
                rank_loss_2 = torch.zeros_like(rank_loss_1)  # [B]
            elif use_mp:
                rank_loss_1 = torch.zeros(B, device=score_pp.device)  # [B]
                rank_loss_2 = F.softplus((self.m2 + score_pp - score_mp) / self.tau)  # [B]
        
        rank_loss = torch.mean(rank_loss_1 + rank_loss_2)
        
        # ============================
        # ============================
        if self.use_similarity:
            if use_pm and use_mp:
                gap1 = (score_pp - score_pm).mean().item()
                gap2 = (score_pm - score_mp).mean().item()
                sat1 = (score_pp > score_pm + self.m1).float().mean().item()
                sat2 = (score_pm > score_mp + self.m2).float().mean().item()
            elif use_pm:
                gap1 = (score_pp - score_pm).mean().item()
                gap2 = 0.0  # 没有第二级
                sat1 = (score_pp > score_pm + self.m1).float().mean().item()
                sat2 = 1.0  # 第二级约束自动满足
            elif use_mp:
                gap1 = 0.0  # 没有第一级
                gap2 = (score_pp - score_mp).mean().item()
                sat1 = 1.0  # 第一级约束自动满足
                sat2 = (score_pp > score_mp + self.m2).float().mean().item()
        else:
            if use_pm and use_mp:
                gap1 = (score_pm - score_pp).mean().item()
                gap2 = (score_mp - score_pm).mean().item()
                sat1 = (score_pp < score_pm - self.m1).float().mean().item()
                sat2 = (score_pm < score_mp - self.m2).float().mean().item()
            elif use_pm:
                gap1 = (score_pm - score_pp).mean().item()
                gap2 = 0.0
                sat1 = (score_pp < score_pm - self.m1).float().mean().item()
                sat2 = 1.0
            elif use_mp:
                gap1 = 0.0
                gap2 = (score_mp - score_pp).mean().item()
                sat1 = 1.0
                sat2 = (score_pp < score_mp - self.m2).float().mean().item()
        
        return rank_loss, {
            'score_pp_mean': score_pp.mean().item(),
            'score_pm_mean': score_pm.mean().item() if use_pm else 0.0,
            'score_mp_mean': score_mp.mean().item() if use_mp else 0.0,
            'rank_loss_1': rank_loss_1.mean().item(),
            'rank_loss_2': rank_loss_2.mean().item(),
            'gap1': gap1,
            'gap2': gap2,
            'sat1': sat1,
            'sat2': sat2
        }


class RankBasedLossProto(nn.Module):
    
    def __init__(self, num_classes=236, 
                 m1=0.1, m2=0.2, beta=1.0, tau=0.1, 
                 K=16, K_prime=16,
                 use_similarity=True, scale=20.0):
        super(RankBasedLossProto, self).__init__()
        
        self.num_classes = num_classes
        self.K = K
        self.K_prime = K_prime
        self.use_similarity = use_similarity
        self.scale = scale
        
        self.rank_loss_fn = RankLossProto(
            m1=m1, m2=m2, beta=beta, tau=tau, 
            K=K, K_prime=K_prime,
            use_similarity=use_similarity, scale=scale
        )
    
    def forward(self, fused_feat_pp, fused_feat_pm, fused_feat_mp, prototype, labels):
        rank_loss, rank_stats = self.rank_loss_fn(
            fused_feat_pp, fused_feat_pm, fused_feat_mp, prototype, labels
        )
        
        loss_dict = {
            'rank_loss': rank_loss.item(),  # 未加权的排序损失
            **rank_stats
        }
        
        return rank_loss, loss_dict
