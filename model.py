import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

dinov3_path = "/data0/shiwei/shiwei/OSR/dinov3-main"
sys.path.append(dinov3_path)


class DINOv3Backbone(nn.Module):
    def __init__(self, pretrained_path=None, device='cuda', logger=None):
        super(DINOv3Backbone, self).__init__()
        
        self.logger = logger
        
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.dinov3_model = torch.hub.load(
            dinov3_path,
            'dinov3_vits16',
            pretrained=False,
            source='local'
        )
        
        if pretrained_path and os.path.exists(pretrained_path):
            if os.path.isdir(pretrained_path):
                if self.logger is not None:
                    self.logger.warning(f"pretrained_path is a directory, not a file: {pretrained_path}")
                else:
                    print(f"pretrained_path is a directory, not a file: {pretrained_path}")
            else:
                if self.logger is not None:
                    self.logger.info(f"Loading DINOv3 pretrained weights from: {pretrained_path}")
                else:
                    print(f"Loading DINOv3 pretrained weights from: {pretrained_path}")
                device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device)
                if torch.cuda.is_available() and device_type == 'cuda':
                    dinov3_weights = torch.load(pretrained_path, map_location='cuda')
                else:
                    dinov3_weights = torch.load(pretrained_path, map_location='cpu')
                
                self.dinov3_model.load_state_dict(dinov3_weights, strict=False)
                if self.logger is not None:
                    self.logger.info("✓ DINOv3 weights loaded successfully")
                else:
                    print("✓ DINOv3 weights loaded successfully")
        self._setup_training_layers()
    
    def _setup_training_layers(self):
        for param in self.dinov3_model.parameters():
            param.requires_grad = False
        
        if hasattr(self.dinov3_model, 'blocks') and isinstance(self.dinov3_model.blocks, nn.ModuleList):
            total_blocks = len(self.dinov3_model.blocks)
            num_trainable_blocks = 4
            start_idx = total_blocks - num_trainable_blocks
            
            for i in range(start_idx, total_blocks):
                for param in self.dinov3_model.blocks[i].parameters():
                    param.requires_grad = True
                if self.logger is not None:
                    self.logger.info(f"Unfreezing DINOv3 block {i+1}/{total_blocks} (index {i})")
                else:
                    print(f"Unfreezing DINOv3 block {i+1}/{total_blocks} (index {i})")
        else:
            if self.logger is not None:
                self.logger.warning("Cannot find DINOv3 blocks; keep backbone frozen.")
            else:
                print("Warning: cannot find DINOv3 blocks; keep backbone frozen.")
        
        if self.logger is not None:
            self.logger.info("Backbone setup: last 4 blocks trainable, first 8 frozen.")
        else:
            print("Backbone setup: last 4 blocks trainable, first 8 frozen.")
    
    def forward(self, x, return_patch_tokens=False):
        has_trainable_params = any(p.requires_grad for p in self.dinov3_model.parameters())
        
        if has_trainable_params:
            features = self.dinov3_model.forward_features(x)
        else:
            with torch.no_grad():
                features = self.dinov3_model.forward_features(x)
        
        if isinstance(features, dict):
            if 'x_norm_clstoken' in features:
                cls_features = features['x_norm_clstoken']
            elif 'cls_token' in features:
                cls_features = features['cls_token']
            else:
                cls_features = None
            
            if 'x_norm_patchtokens' in features:
                patch_tokens = features['x_norm_patchtokens']
            elif 'patch_tokens' in features:
                patch_tokens = features['patch_tokens']
            else:
                patch_tokens = None
            
            if cls_features is None or patch_tokens is None:
                keys = list(features.keys())
                if keys:
                    first_feat = features[keys[0]]
                    if len(first_feat.shape) == 3:
                        cls_features = first_feat[:, 0, :]
                        patch_tokens = first_feat[:, 1:, :]
                    elif len(first_feat.shape) == 2:
                        cls_features = first_feat
                        patch_tokens = None
        else:
            if len(features.shape) == 3:
                cls_features = features[:, 0, :]
                patch_tokens = features[:, 1:, :]
            elif len(features.shape) == 2:
                cls_features = features
                patch_tokens = None
            else:
                raise ValueError(f"Unexpected DINOv3 feature shape: {features.shape}")
        
        if return_patch_tokens:
            return cls_features, patch_tokens
        else:
            return cls_features


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, num_layers=2, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        self.fuse_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers * 4)
        ])
        
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, fg_patch_tokens, bg_patch_tokens, fg_cls=None, bg_cls=None):
        """
        Fuse foreground and background features.
        Args:
            fg_patch_tokens: [B, N_fg, 384]
            bg_patch_tokens: [B, N_bg, 384]
            fg_cls: [B, 384] optional
            bg_cls: [B, 384] optional
        Returns:
            fused_features: [B, 384]
        """
        B = fg_patch_tokens.shape[0]
        
        fuse_tokens = self.fuse_token.expand(B, -1, -1)  # [B, 1, 384]
        
        x = fuse_tokens  # [B, 1, 384]
        norm_idx = 0
        
        for i in range(self.num_layers):
            residual = x
            x = self.norm_layers[norm_idx](x)
            attn_out, _ = self.cross_attention_layers[i](x, fg_patch_tokens, fg_patch_tokens)
            x = residual + attn_out
            norm_idx += 1
            
            residual = x
            x = self.norm_layers[norm_idx](x)
            ffn_out = self.ffn_layers[i](x)
            x = residual + ffn_out
            norm_idx += 1
            
            residual = x
            x = self.norm_layers[norm_idx](x) if norm_idx < len(self.norm_layers) else x
            attn_out, _ = self.cross_attention_layers[i](x, bg_patch_tokens, bg_patch_tokens)
            x = residual + attn_out
            norm_idx += 1
            residual = x
            x = self.norm_layers[norm_idx](x) if norm_idx < len(self.norm_layers) else x
            ffn_out = self.ffn_layers[i](x)
            x = residual + ffn_out
            norm_idx += 1
        fused_fuse = x.squeeze(1)
        if fg_cls is not None and bg_cls is not None:
            concat_features = torch.cat([fused_fuse, fg_cls, bg_cls], dim=1)
            fused_features = self.final_fusion(torch.cat([fused_fuse, (fg_cls + bg_cls) / 2], dim=1))
        else:
            if bg_cls is not None:
                concat_features = torch.cat([fused_fuse, bg_cls], dim=1)
            else:
                concat_features = torch.cat([fused_fuse, fused_fuse], dim=1)
            fused_features = self.final_fusion(concat_features)  # [B, 384]
        
        return fused_features


class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, scale=20.0, margin=0.0):
        super(CosineClassifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.scale = scale
        self.margin = margin
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, labels=None):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        logits = self.scale * x_norm @ w_norm.t()
        if labels is not None and self.margin > 0:
            correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            logits = logits.clone()
            logits.scatter_(1, labels.unsqueeze(1), correct_logits.unsqueeze(1) - self.scale * self.margin)
        
        return logits


class FusionClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained_path=None, 
                 fusion_layers=4, fusion_heads=6, dropout=0.1, device='cuda', logger=None,
                 cosine_scale=20.0, cosine_margin=0.0):
        super(FusionClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        self.backbone = DINOv3Backbone(pretrained_path=pretrained_path, device=device, logger=logger        )
        
        self.fusion_layer = CrossAttentionFusion(
            embed_dim=384,
            num_heads=fusion_heads,
            num_layers=fusion_layers,
            dropout=dropout
        )
        
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        self.classifier = CosineClassifier(feat_dim=384, num_classes=num_classes, 
                                           scale=cosine_scale, margin=cosine_margin        )
        
        self._init_proj_weights()
    
    def extract_all_features(self, foreground, background):
        """
        Extract three-level features for visualization.
        Args:
            foreground: [B, 3, 224, 224]
            background: [B, 3, 224, 224]
        Returns:
            feat_proj: [B, 384], fg_cls: [B, 384], bg_cls: [B, 384]
        """
        B = foreground.size(0)
        
        x = torch.cat([foreground, background], dim=0)  # [2B, 3, 224, 224]
        
        all_cls, all_patch_tokens = self.backbone(x, return_patch_tokens=True)  # [2B, 384], [2B, N, 384]
        
        fg_cls = all_cls[:B]  # [B, 384]
        bg_cls = all_cls[B:]  # [B, 384]
        fg_patch_tokens = all_patch_tokens[:B]  # [B, N, 384]
        bg_patch_tokens = all_patch_tokens[B:]  # [B, N, 384]
        
        fused_features = self.fusion_layer(
            fg_patch_tokens, bg_patch_tokens, 
            fg_cls=fg_cls, bg_cls=bg_cls
        )  # [B, 384]
        
        feat_proj = self.feat_proj(fused_features)  # [B, 384]
        
        return feat_proj, fg_cls, bg_cls
    
    def _init_proj_weights(self):
        """Initialize projection layer weights."""
        for m in self.feat_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
    
    def forward(self, foreground, background, multi_instance=False, return_features=False, return_all_features=False, labels=None):
        """
        Forward pass. Returns logits and optionally fused/feature tensors.
        """
        B = foreground.size(0)
        
        x = torch.cat([foreground, background], dim=0)  # [2B, 3, 224, 224]
        
        all_cls, all_patch_tokens = self.backbone(x, return_patch_tokens=True)  # [2B, 384], [2B, N, 384]
        
        fg_cls = all_cls[:B]  # [B, 384]
        bg_cls = all_cls[B:]  # [B, 384]
        fg_patch_tokens = all_patch_tokens[:B]  # [B, N, 384]
        bg_patch_tokens = all_patch_tokens[B:]  # [B, N, 384]
        
        if multi_instance:
            fg_patch_expanded = fg_patch_tokens.unsqueeze(1).expand(B, B, -1, -1)  # [B, B, N, 384]
            fg_cls_expanded = fg_cls.unsqueeze(1).expand(B, B, -1)  # [B, B, 384]
            
            bg_patch_expanded = bg_patch_tokens.unsqueeze(0).expand(B, B, -1, -1)  # [B, B, N, 384]
            bg_cls_expanded = bg_cls.unsqueeze(0).expand(B, B, -1)  # [B, B, 384]
            
            B_squared = B * B
            fg_patch_flat = fg_patch_expanded.contiguous().view(B_squared, -1, 384)  # [B*B, N, 384]
            bg_patch_flat = bg_patch_expanded.contiguous().view(B_squared, -1, 384)  # [B*B, N, 384]
            fg_cls_flat = fg_cls_expanded.contiguous().view(B_squared, 384)  # [B*B, 384]
            bg_cls_flat = bg_cls_expanded.contiguous().view(B_squared, 384)  # [B*B, 384]
            
            fused_features_flat = self.fusion_layer(
                fg_patch_flat, bg_patch_flat,
                fg_cls=fg_cls_flat, bg_cls=bg_cls_flat
            )  # [B*B, 384]
            
            fused_proj_flat = self.feat_proj(fused_features_flat)  # [B*B, 384]
            
            if labels is not None:
                labels_flat = labels.unsqueeze(1).expand(B, B).contiguous().view(B_squared)  # [B*B]
                logits_flat = self.classifier(fused_proj_flat, labels=labels_flat)  # [B*B, num_classes]
            else:
                logits_flat = self.classifier(fused_proj_flat)  # [B*B, num_classes]
            logits = logits_flat.view(B, B, self.num_classes)  # [B, B, num_classes]
            
            if return_all_features:
                fused_proj = fused_proj_flat.view(B, B, 384)  # [B, B, 384]
                fg_cls_expanded = fg_cls.unsqueeze(1).expand(B, B, -1)  # [B, B, 384]
                bg_cls_expanded = bg_cls.unsqueeze(0).expand(B, B, -1)  # [B, B, 384]
                return logits, fused_proj, fg_cls_expanded, bg_cls_expanded  # [B, B, num_classes], [B, B, 384], [B, B, 384], [B, B, 384]
            elif return_features:
                fused_proj = fused_proj_flat.view(B, B, 384)  # [B, B, 384]
                return logits, fused_proj  # [B, B, num_classes], [B, B, 384]
            else:
                return logits
            
        else:
            fused_features = self.fusion_layer(
                fg_patch_tokens, bg_patch_tokens, 
                fg_cls=fg_cls, bg_cls=bg_cls
            )  # [B, 384]
            
            feat_proj = self.feat_proj(fused_features)  # [B, 384]
            
            logits = self.classifier(feat_proj, labels=labels)  # [B, num_classes]
            
            if return_all_features:
                return logits, feat_proj, fg_cls, bg_cls  # [B, num_classes], [B, 384], [B, 384], [B, 384]
            elif return_features:
                return logits, feat_proj, bg_cls  # [B, num_classes], [B, 384], [B, 384]
            else:
                return logits


