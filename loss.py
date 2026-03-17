import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_log_odds(logits, target_labels, eps=1e-8):
    """
    Compute one-vs-rest log-odds for target classes.
    """
    target_logits = logits.gather(1, target_labels.unsqueeze(1)).squeeze(1)
    one_hot_mask = F.one_hot(target_labels, num_classes=logits.size(1)).bool()
    masked_logits = logits.masked_fill(one_hot_mask, float("-inf"))
    log_sum_exp_non_target = torch.logsumexp(masked_logits, dim=1)
    log_odds = target_logits - log_sum_exp_non_target
    
    return log_odds


class RankLoss(nn.Module):
    def __init__(self, m1=1.0, m2=1.5, beta=3.0, tau=1.0, K=3, K_prime=3):
        super(RankLoss, self).__init__()
        self.m1 = m1
        self.m2 = m2
        self.beta = beta
        self.tau = tau
        self.K = K
        self.K_prime = K_prime
    
    def forward(self, logits_pp, logits_pm, logits_mp, labels):
        B = labels.size(0)
        N_pm = logits_pm.size(1)
        N_mp = logits_mp.size(1)
        
        r_pp = compute_log_odds(logits_pp, labels)
        logits_pm_flat = logits_pm.view(-1, logits_pm.size(-1))
        labels_pm_flat = labels.unsqueeze(1).expand(B, N_pm).contiguous().view(-1)
        r_pm_all = compute_log_odds(logits_pm_flat, labels_pm_flat)
        r_pm_all = r_pm_all.view(B, N_pm)
        if N_pm > self.K:
            r_pm_topk, _ = torch.topk(r_pm_all, k=self.K, dim=1)
            r_pm = r_pm_topk.mean(dim=1)
        else:
            r_pm, _ = torch.max(r_pm_all, dim=1)
        
        logits_mp_flat = logits_mp.view(-1, logits_mp.size(-1))
        labels_mp_flat = labels.unsqueeze(1).expand(B, N_mp).contiguous().view(-1)
        r_mp_all = compute_log_odds(logits_mp_flat, labels_mp_flat)
        r_mp_all = r_mp_all.view(B, N_mp)
        if N_mp > self.K_prime:
            r_mp_topk, _ = torch.topk(r_mp_all, k=self.K_prime, dim=1)
            r_mp = r_mp_topk.mean(dim=1)
        else:
            r_mp, _ = torch.max(r_mp_all, dim=1)
        
        rank_loss_1 = F.softplus((self.m1 + r_pm - r_pp) / self.tau)
        rank_loss_2 = self.beta * F.softplus((self.m2 + r_mp - r_pm) / self.tau)
        rank_loss = torch.mean(rank_loss_1 + rank_loss_2)

        gap1 = (r_pp - r_pm).mean().item()
        gap2 = (r_pm - r_mp).mean().item()
        sat1 = (r_pp > r_pm + self.m1).float().mean().item()
        sat2 = (r_pm > r_mp + self.m2).float().mean().item()
        
        return rank_loss, {
            'r_pp_mean': r_pp.mean().item(),
            'r_pm_mean': r_pm.mean().item(),
            'r_mp_mean': r_mp.mean().item(),
            'rank_loss_1': rank_loss_1.mean().item(),
            'rank_loss_2': rank_loss_2.mean().item(),
            'gap1': gap1,
            'gap2': gap2,
            'sat1': sat1,
            'sat2': sat2
        }


class RankBasedLoss(nn.Module):
    def __init__(self, num_classes=236, 
                 m1=1.0, m2=1.5, beta=3.0, tau=1.0, 
                 K=3, K_prime=3,
                 lambda_rank=0.5):
        super(RankBasedLoss, self).__init__()
        
        self.num_classes = num_classes
        self.K = K
        self.K_prime = K_prime
        self.lambda_rank = lambda_rank
        
        self.rank_loss_fn = RankLoss(m1=m1, m2=m2, beta=beta, tau=tau, K=K, K_prime=K_prime)
        
        self.ce_loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, logits_pp, logits_pp_rank, logits_pm_rank, logits_mp_rank, labels):
        ce_loss = self.ce_loss_fn(logits_pp, labels)
        rank_loss, rank_stats = self.rank_loss_fn(logits_pp_rank, logits_pm_rank, logits_mp_rank, labels)
        w_rank = self.lambda_rank * rank_loss
        total_loss = ce_loss + w_rank
        loss_dict = {
            'ce_loss': ce_loss.item(),
            'rank_loss': rank_loss.item(),
            'total_loss': total_loss.item(),
            'w_rank': w_rank.item(),
            **rank_stats
        }
        
        return total_loss, loss_dict

