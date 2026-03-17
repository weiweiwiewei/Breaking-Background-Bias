
import torch
import torch.nn.functional as F
import warnings
from tqdm import tqdm
from loss_proto import RankBasedLossProto


def train_epoch_proto(model, train_loader, rank_loss_fn, optimizer, device, epoch,
                      gradient_accumulation_steps=1, 
                      warmup_epochs=1, warmup_ramp_epochs=2,
                      lambda_rank=1.0):
    # ============================
    # ============================
    model.train()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_rank_loss = 0.0
    running_gap1 = 0.0
    running_gap2 = 0.0
    running_sat1 = 0.0
    running_sat2 = 0.0
    correct = 0
    total = 0
    effective_batches = 0
    accum_counter = 0
    
    # ============================
    # ============================
    use_rank_loss = epoch > warmup_epochs
    
    if epoch <= warmup_epochs:
        weight_scale = 0.0
    elif epoch <= warmup_epochs + warmup_ramp_epochs:
        weight_scale = (epoch - warmup_epochs) / warmup_ramp_epochs
    else:
        weight_scale = 1.0
    
    current_lambda_rank = lambda_rank * weight_scale
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train-Proto]', ncols=120)
    optimizer.zero_grad()
    
    for batch_idx, (foreground, background, labels) in enumerate(pbar):
        foreground = foreground.to(device, non_blocking=True)
        background = background.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        B = labels.size(0)
        D = 384  # 融合特征维度
        
        if B < 2:
            continue
        
        # ============================
        # ============================
        out = model(
            foreground, background,
            multi_instance=True,
            return_features=True  # 需要返回融合特征
        )
        logits_all = out[0] if isinstance(out, (tuple, list)) else out  # [B, B, C]
        if isinstance(out, (tuple, list)) and len(out) > 1:
            fused_feat_all = out[1]  # [B, B, 384]
        else:
            raise ValueError("模型必须返回融合特征")
        
        # ============================
        # ============================
        idx = torch.arange(B, device=device)
        
        fused_feat_pp = fused_feat_all[idx, idx]  # [B, 384]
        logits_pp = logits_all[idx, idx]  # [B, C]
        
        num_unique_labels = torch.unique(labels).numel()
        batch_is_single_class = (num_unique_labels == 1)
        
        if not use_rank_loss or batch_is_single_class:
            ce_loss = F.cross_entropy(logits_pp, labels)
            loss = ce_loss / gradient_accumulation_steps
            loss.backward()
            accum_counter += 1
            effective_batches += 1
            
            if accum_counter % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += ce_loss.item()
            running_ce_loss += ce_loss.item()  # 记录 CE 损失
            _, predicted = torch.max(logits_pp.data, 1)
            total += B
            correct += (predicted == labels).sum().item()
            
            acc_pct = 100.0 * correct / total if total > 0 else 0.0
            warmup_status = 'Warmup' if not use_rank_loss else 'SingleClass'
            pbar.set_postfix({
                'L': f'{ce_loss.item():.4f}',
                'CE': f'{ce_loss.item():.4f}',
                'Acc': f'{acc_pct:.1f}%',
                'W': warmup_status
            })
            continue
        
        off_diagonal_mask = ~torch.eye(B, dtype=torch.bool, device=device)  # [B, B]
        
        K = rank_loss_fn.K  # 从 rank_loss_fn 获取 K
        
        if K > 0:
            labels_i = labels.unsqueeze(1)  # [B, 1]
            labels_j = labels.unsqueeze(0)  # [1, B]
            cross_class_offdiag = (labels_i != labels_j) & off_diagonal_mask  # [B, B]，跨类且非对角线
            
            fused_feat_pm_list = []
            pm_fallback = 0  # 记录回退次数
            
            for i in range(B):
                cand = torch.where(cross_class_offdiag[i])[0]  # j candidates
                n = cand.numel()
                
                if n == 0:
                    pm_fallback += 1
                    cand2 = torch.where(off_diagonal_mask[i])[0]  # [B-1]
                    if cand2.numel() >= K:
                        sel = cand2[torch.randperm(cand2.numel(), device=device)[:K]]
                    else:
                        sel = cand2[torch.randint(0, cand2.numel(), (K,), device=device)]
                else:
                    if n >= K:
                        sel = cand[torch.randperm(n, device=device)[:K]]
                    else:
                        sel = cand[torch.randint(0, n, (K,), device=device)]
                
                fused_feat_pm_list.append(fused_feat_all[i, sel])  # [K, D]
            
            fused_feat_pm = torch.stack(fused_feat_pm_list, dim=0)  # [B, K, D]
            
            if pm_fallback > 0 and batch_idx == 0:
                warnings.warn(
                    f"[Epoch {epoch}] Batch contains samples with same labels. "
                    f"Falling back to j!=i for {pm_fallback}/{B} samples in pm construction. "
                    f"This may happen if batch is too small or contains many samples from the same class.",
                    UserWarning
                )
        else:
            D = fused_feat_all.shape[-1]
            fused_feat_pm = torch.empty(B, 0, D, device=device, dtype=fused_feat_all.dtype)  # [B, 0, D]
        
        K_prime = rank_loss_fn.K_prime  # 从 rank_loss_fn 获取 K_prime
        
        if K_prime > 0:
            labels_expanded_i = labels.unsqueeze(1)  # [B, 1]
            labels_expanded_k = labels.unsqueeze(0)  # [1, B]
            cross_class_mask = (labels_expanded_i != labels_expanded_k)  # [B, B]，跨类 mask
            
            fused_feat_mp_list = []
            fallback_count = 0  # 记录回退次数
            
            for i in range(B):
                cross_class_candidates = torch.where(cross_class_mask[i])[0]  # [N_candidates]
                N_candidates = cross_class_candidates.size(0)
                
                if N_candidates == 0:
                    fallback_count += 1
                    off_diag_candidates = torch.where(off_diagonal_mask[i])[0]  # [B-1]
                    if off_diag_candidates.size(0) == 0:
                        raise ValueError(f"Batch size too small (B={B}) to construct mp samples")
                    
                    if off_diag_candidates.size(0) >= K_prime:
                        selected_indices = torch.randperm(off_diag_candidates.size(0), device=device)[:K_prime]
                        selected_k = off_diag_candidates[selected_indices]
                    else:
                        selected_k = off_diag_candidates[torch.randint(0, off_diag_candidates.size(0), (K_prime,), device=device)]
                    
                    mp_features_i = fused_feat_all[selected_k, selected_k]  # [K_prime, D]
                else:
                    if N_candidates >= K_prime:
                        selected_indices = torch.randperm(N_candidates, device=device)[:K_prime]
                        selected_k = cross_class_candidates[selected_indices]
                    else:
                        selected_k = cross_class_candidates[torch.randint(0, N_candidates, (K_prime,), device=device)]
                
                    mp_features_i = fused_feat_all[selected_k, selected_k]  # [K_prime, D]
                
                fused_feat_mp_list.append(mp_features_i)
            
            fused_feat_mp = torch.stack(fused_feat_mp_list, dim=0)  # [B, K_prime, D]
            
            if fallback_count > 0 and batch_idx == 0:
                warnings.warn(
                    f"[Epoch {epoch}] Some samples have no cross-class candidates. "
                    f"Falling back to k!=i for {fallback_count}/{B} samples in mp construction. "
                    f"This should be rare if batch is diverse.",
                    UserWarning
                )
        else:
            D = fused_feat_all.shape[-1]
            fused_feat_mp = torch.empty(B, 0, D, device=device, dtype=fused_feat_all.dtype)  # [B, 0, D]
        
        # ============================
        # ============================
        
        ce_loss = F.cross_entropy(logits_pp, labels)
        
        assert hasattr(model, "classifier"), "model needs classifier attr for prototype rank loss"
        prototype = model.classifier.weight  # [C, 384]
        prototype_rank = prototype.detach()  # stop-grad for rank only，不影响 CE 更新 head
        
        rank_loss, rank_loss_dict = rank_loss_fn(
            fused_feat_pp, fused_feat_pm, fused_feat_mp, prototype_rank, labels
        )
        
        total_loss = ce_loss + current_lambda_rank * rank_loss
        
        # ============================
        # ============================
        loss_value = total_loss.item()
        (total_loss / gradient_accumulation_steps).backward()
        accum_counter += 1
        effective_batches += 1
        
        if accum_counter % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # ============================
        # ============================
        running_loss += loss_value
        running_ce_loss += ce_loss.item()
        running_rank_loss += rank_loss_dict['rank_loss']  # 未加权的 rank_loss
        
        if 'gap1' in rank_loss_dict:
            running_gap1 += rank_loss_dict['gap1']
            running_gap2 += rank_loss_dict['gap2']
            running_sat1 += rank_loss_dict['sat1']
            running_sat2 += rank_loss_dict['sat2']
        
        _, predicted = torch.max(logits_pp.data, 1)
        total += B
        correct += (predicted == labels).sum().item()
        
        acc_pct = 100.0 * correct / total if total > 0 else 0.0
        warmup_status = f'Ramp({weight_scale:.2f})' if weight_scale < 1.0 else 'Full'
        w_rank_display = (current_lambda_rank * rank_loss).item()  # 当前加权后的排序损失
        
        pbar.set_postfix({
            'L': f'{loss_value:.4f}',  # 总损失
            'CE': f'{ce_loss.item():.4f}',  # CE 损失
            'wR': f'{w_rank_display:.4f}',  # 加权后的排序损失（考虑 warmup/ramp）
            'Acc': f'{acc_pct:.1f}%',
            's++': f'{rank_loss_dict.get("score_pp_mean", 0.0):.2f}',
            's+-': f'{rank_loss_dict.get("score_pm_mean", 0.0):.2f}',
            's-+': f'{rank_loss_dict.get("score_mp_mean", 0.0):.2f}',
            'W': warmup_status
        })
    
    # ============================
    # ============================
    if accum_counter > 0 and accum_counter % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # ============================
    # ============================
    denom = max(effective_batches, 1)
    epoch_loss = running_loss / denom
    epoch_ce_loss = running_ce_loss / denom
    epoch_rank_loss = running_rank_loss / denom if use_rank_loss else 0.0
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0
    
    epoch_gap1 = running_gap1 / denom if use_rank_loss else 0.0
    epoch_gap2 = running_gap2 / denom if use_rank_loss else 0.0
    epoch_sat1 = running_sat1 / denom if use_rank_loss else 0.0
    epoch_sat2 = running_sat2 / denom if use_rank_loss else 0.0
    
    loss_dict_summary = {
        'ce_loss': epoch_ce_loss,
        'rank_loss': epoch_rank_loss,
        'total_loss': epoch_loss,
        'gap1': epoch_gap1,
        'gap2': epoch_gap2,
        'sat1': epoch_sat1,
        'sat2': epoch_sat2
    }
    
    return epoch_loss, epoch_acc, loss_dict_summary
