import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from torchvision import transforms

from model import FusionClassifier
from dataset import ImageNetAnimalFusionDataset
from utils import setup_logging, set_seed
from trainer_proto import train_epoch_proto
from evaluator import (
    compute_fuse_prototypes, 
    evaluate, 
    evaluate_with_prototype,
    evaluate_ood,
    evaluate_open_set_recognition
)
from loss_proto import RankBasedLossProto


def main():
    parser = argparse.ArgumentParser(description='DINOv3 fusion model training')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['animal', 'object', 'other', 'nico', 'vlcs'],
        default='other',
        help='Dataset split: animal/object/other (ImageNet), or nico/vlcs (domain generalization data).'
    )
    parser.add_argument(
        '--imagenet_root',
        type=str,
        default=None,
        help='Root directory of ImageNet (containing train/val); auto-set when --dataset is used.'
    )
    parser.add_argument(
        '--id_class_list',
        type=str,
        default=None,
        help='Path to ID class list file; auto-set when --dataset is used.'
    )
    parser.add_argument(
        '--ood_class_list',
        type=str,
        default=None,
        help='Path to OOD class list file; auto-set when --dataset is used.'
    )
    parser.add_argument(
        '--pretrained_path',
        type=str,
        default='/data0/shiwei/shiwei/OSR/dinov3_ckp/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        help='Path to DINOv3 pretrained weights.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=236,
        help='Number of ID classes (default: 236 for ImageNet-animal).'
    )
    parser.add_argument(
        '--fusion_layers',
        type=int,
        default=2,
        help='Number of cross-attention fusion layers (recommended: 2–4).'
    )
    parser.add_argument(
        '--fusion_heads',
        type=int,
        default=6,
        help='Number of attention heads in Transformer fusion.'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate.'
    )
    parser.add_argument(
        '--cosine_scale',
        type=float,
        default=20.0,
        help='Scale parameter for cosine classifier (default: 20.0).'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=48,
        help='Batch size for training.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5,
        help='Base learning rate (default: 1e-5).'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.05,
        help='Weight decay for AdamW optimizer (default: 0.05).'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=2,
        help='Gradient accumulation steps. Effective batch size = batch_size × gradient_accumulation_steps.'
    )
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='cosine',
        choices=['none', 'cosine', 'reduce_on_plateau'],
        help='Learning rate scheduler: none, cosine, or reduce_on_plateau (default: cosine).'
    )
    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=5,
        help='Number of warmup epochs (default: 5; set 0 to disable warmup).'
    )
    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=None,
        help='Learning rate at the end of warmup (default: same as learning_rate).'
    )
    parser.add_argument(
        '--scheduler_patience',
        type=int,
        default=5,
        help='Patience for ReduceLROnPlateau scheduler (default: 5).'
    )
    parser.add_argument(
        '--scheduler_factor',
        type=float,
        default=0.5,
        help='Factor for ReduceLROnPlateau scheduler (default: 0.5).'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--eval_start_epoch',
        type=int,
        default=50,
        help='Start evaluation from this epoch index.'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of data loader workers.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use: cuda or cpu.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed.'
    )
    parser.add_argument(
        '--ood_batch_size',
        type=int,
        default=64,
        help='Batch size for OOD evaluation.'
    )
    parser.add_argument(
        '--ood_percentile',
        type=float,
        default=95.0,
        help='Percentile to set OOD threshold (default: 95.0, suggested: 90–99).'
    )
    parser.add_argument(
        '--no_index_cache',
        action='store_true',
        help='Disable index file caching.'
    )
    parser.add_argument(
        '--use_rank_loss',
        action='store_true',
        default=True,
        help='Enable rank-based loss (default: True).'
    )
    parser.add_argument(
        '--no_rank_loss',
        dest='use_rank_loss',
        action='store_false',
        help='Disable rank-based loss (opposite of --use_rank_loss).'
    )
    parser.add_argument(
        '--rank_m1',
        type=float,
        default=1.0,
        help='First margin m1 in rank loss.'
    )
    parser.add_argument(
        '--rank_m2',
        type=float,
        default=1.5,
        help='Second margin m2 in rank loss.'
    )
    parser.add_argument(
        '--rank_beta',
        type=float,
        default=1.0,
        help='Weight beta for the third term in rank loss.'
    )
    parser.add_argument(
        '--rank_tau',
        type=float,
        default=0.25,
        help='Temperature tau in rank loss.'
    )
    parser.add_argument(
        '--rank_K',
        type=int,
        default=0,
        help='Number of background samples K.'
    )
    parser.add_argument(
        '--rank_K_prime',
        type=int,
        default=4,
        help='Number of foreground samples K prime.'
    )
    parser.add_argument(
        '--lambda_rank',
        type=float,
        default=0,
        help='Weight for rank loss: total_loss = ce_loss + lambda_rank * rank_loss.'
    )
    parser.add_argument(
        '--use_distance',
        action='store_true',
        help='Use L2 distance for ranking instead of cosine similarity.'
    )

    parser.add_argument(
        '--rank_warmup_epochs',
        type=int,
        default=5,
        help='Epochs to warm up rank loss (use CE only in early epochs).'
    )
    parser.add_argument(
        '--rank_warmup_ramp_epochs',
        type=int,
        default=5,
        help='Epochs to linearly ramp up rank loss weight.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/data0/shiwei/shiwei/OSR/Breaking-Background_Bias/checkpoints',
        help='Directory to save model checkpoints.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/data0/shiwei/shiwei/OSR/Breaking-Background_Bias/logs',
        help='Directory to save training logs.'
    )
    
    args = parser.parse_args()
    
    index_dir = '/data0/shiwei/shiwei/OSR/imagenet_animial_split'
    imagenet_processed_data = '/data0/shiwei/shiwei/OSR/dataset/Processed_Data/ImageNet'
    
    if args.dataset is not None:
        if args.dataset == 'animal':
            if args.imagenet_root is None:
                args.imagenet_root = imagenet_processed_data
            if args.id_class_list is None:
                args.id_class_list = os.path.join(index_dir, 'imagenet_animal_id_classes.txt')
            if args.ood_class_list is None:
                args.ood_class_list = os.path.join(index_dir, 'imagenet_animal_ood_classes.txt')
        elif args.dataset == 'object':
            if args.imagenet_root is None:
                args.imagenet_root = imagenet_processed_data
            if args.id_class_list is None:
                args.id_class_list = os.path.join(index_dir, 'imagenet_object_id_classes.txt')
            if args.ood_class_list is None:
                args.ood_class_list = os.path.join(index_dir, 'imagenet_object_ood_classes.txt')
        elif args.dataset == 'other':
            if args.imagenet_root is None:
                args.imagenet_root = imagenet_processed_data
            if args.id_class_list is None:
                args.id_class_list = os.path.join(index_dir, 'imagenet_other_id_classes.txt')
            if args.ood_class_list is None:
                args.ood_class_list = os.path.join(index_dir, 'imagenet_other_ood_classes.txt')
        elif args.dataset == 'nico':
            if args.imagenet_root is None:
                args.imagenet_root = '/data0/shiwei/shiwei/OSR/dataset/Processed_Data/data_NICO'
            if args.id_class_list is None:
                default_id_file = os.path.join('/data0/shiwei/shiwei/OSR/RWOSR1/data_NICO_RWOSR', 'id_classes.txt')
                if os.path.exists(default_id_file):
                    args.id_class_list = default_id_file
                else:
                    args.id_class_list = None
            if args.ood_class_list is None:
                default_ood_file = os.path.join('/data0/shiwei/shiwei/OSR/RWOSR1/data_NICO_RWOSR', 'ood_classes.txt')
                if os.path.exists(default_ood_file):
                    args.ood_class_list = default_ood_file
                else:
                    args.ood_class_list = None
        elif args.dataset == 'vlcs':
            if args.imagenet_root is None:
                args.imagenet_root = '/data0/shiwei/shiwei/OSR/dataset/Processed_Data/data_VLCS'
            if args.id_class_list is None:
                default_id_file = os.path.join('/data0/shiwei/shiwei/OSR/RWOSR1/data_VLCS_Nico', 'id_classes.txt')
                if os.path.exists(default_id_file):
                    args.id_class_list = default_id_file
                else:
                    args.id_class_list = None
            if args.ood_class_list is None:
                default_ood_file = os.path.join('/data0/shiwei/shiwei/OSR/RWOSR1/data_VLCS_Nico', 'ood_classes.txt')
                if os.path.exists(default_ood_file):
                    args.ood_class_list = default_ood_file
                else:
                    args.ood_class_list = None
    else:
        if args.imagenet_root is None:
            args.imagenet_root = imagenet_processed_data
        if args.id_class_list is None:
            args.id_class_list = os.path.join(index_dir, 'imagenet_animal_id_classes.txt')
        if args.ood_class_list is None:
            args.ood_class_list = os.path.join(index_dir, 'imagenet_animal_ood_classes.txt')
    
    set_seed(args.seed)
    
    def generate_rank_filename(prefix, extension=''):
        """Generate filename from rank loss params (cosine similarity)."""
        dataset_str = f"_{args.dataset}" if args.dataset else ""
        
        def format_float(f):
            """Format float, strip trailing zeros."""
            s = f"{f:.2f}".rstrip('0').rstrip('.')
            return s if s else "0"
        
        filename = (
            f"{prefix}"
            f"{dataset_str}_"
            f"m1-{format_float(args.rank_m1)}_"
            f"m2-{format_float(args.rank_m2)}_"
            f"beta-{format_float(args.rank_beta)}_"
            f"tau-{format_float(args.rank_tau)}_"
            f"K-{args.rank_K}_"
            f"Kp-{args.rank_K_prime}_"
            f"lr-{format_float(args.lambda_rank)}"
        )
        if extension:
            filename += f".{extension}"
        return filename
    
    log_name = generate_rank_filename("train_fusion", "log")
    logger = setup_logging(log_dir=args.log_dir, log_name=log_name)
    logger.info("=" * 80)
    logger.info("DINOv3 fusion model training - ImageNet animal dataset")
    logger.info("=" * 80)
    logger.info(f"  Dataset: {args.dataset if args.dataset else 'default'}")
    logger.info(f"  ImageNet root: {args.imagenet_root}")
    logger.info(f"  ID class list: {args.id_class_list}")
    logger.info(f"  OOD class list: {args.ood_class_list}")
    logger.info(f"  Pretrained path: {args.pretrained_path}")
    logger.info(f"  Num classes: {args.num_classes}")
    logger.info(f"  Fusion layers: {args.fusion_layers}")
    logger.info(f"  Fusion heads: {args.fusion_heads}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(f"  Cosine scale: {args.cosine_scale}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"  LR scheduler: {args.lr_scheduler}")
    logger.info(f"  Warmup epochs: {args.warmup_epochs}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Eval start epoch: {args.eval_start_epoch}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Save dir: {args.save_dir}")
    logger.info(f"  Log dir: {args.log_dir}")
    
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Device: {device} (GPU: {torch.cuda.get_device_name(0)})")
        else:
            raise RuntimeError("CUDA not available. Check: 1) GPU installed 2) CUDA driver 3) PyTorch built with CUDA.")
    else:
        device = torch.device('cpu')
        logger.info(f"Device: {device}")
    
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    use_index_cache = not args.no_index_cache
    
    if args.dataset in ['nico', 'vlcs']:
        train_dataset = ImageNetAnimalFusionDataset(
            imagenet_root=args.imagenet_root,
            class_list_file=args.id_class_list,
            split='train',
            transform=transform_train,
            use_cache=use_index_cache,
            logger=logger
        )
        test_dataset = ImageNetAnimalFusionDataset(
            imagenet_root=args.imagenet_root,
            class_list_file=args.id_class_list,
            split='test',
            transform=transform_eval,
            use_cache=use_index_cache,
            logger=logger
        )
        ood_dataset = ImageNetAnimalFusionDataset(
            imagenet_root=args.imagenet_root,
            class_list_file=args.ood_class_list,
            split='unknow1',
            transform=transform_eval,
            use_cache=use_index_cache,
            logger=logger
        )
    else:
        train_dataset = ImageNetAnimalFusionDataset(
            imagenet_root=args.imagenet_root,
            class_list_file=args.id_class_list,
            split='train',
            transform=transform_train,
            use_cache=use_index_cache,
            logger=logger
        )
        test_dataset = ImageNetAnimalFusionDataset(
            imagenet_root=args.imagenet_root,
            class_list_file=args.id_class_list,
            split='val',
            transform=transform_eval,
            use_cache=use_index_cache,
            logger=logger
        )
        ood_dataset = ImageNetAnimalFusionDataset(
            imagenet_root=args.imagenet_root,
            class_list_file=args.ood_class_list,
            split='val',
            transform=transform_eval,
            use_cache=use_index_cache,
            logger=logger
        )
    
    num_classes = len(train_dataset.class_names)
    if num_classes != args.num_classes:
        logger.warning(f"Dataset num_classes ({num_classes}) != args.num_classes ({args.num_classes}), using {num_classes}")
        args.num_classes = num_classes
    
    drop_last = args.use_rank_loss
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=8 if args.num_workers > 0 else 2,
        drop_last=drop_last
    )
    if drop_last:
        logger.info("Rank mode: DataLoader drop_last=True to avoid small batch.")
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=8 if args.num_workers > 0 else 2
    )
    ood_loader = DataLoader(
        ood_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=8 if args.num_workers > 0 else 2
    )
    
    model = FusionClassifier(
        num_classes=args.num_classes,
        pretrained_path=args.pretrained_path,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        dropout=args.dropout,
        device=device,
        logger=logger,
        cosine_scale=args.cosine_scale,
        cosine_margin=0.0
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay    )
    
    if not args.use_rank_loss:
        logger.warning("--no_rank_loss set but rank loss required; forcing use_rank_loss=True.")
        args.use_rank_loss = True
    
    rank_loss_fn = None
    if args.use_rank_loss:
        rank_loss_fn = RankBasedLossProto(
            num_classes=args.num_classes,
            m1=args.rank_m1,
            m2=args.rank_m2,
            beta=args.rank_beta,
            tau=args.rank_tau,
            K=args.rank_K,
            K_prime=args.rank_K_prime,
            scale=args.cosine_scale
        ).to(device)
        logger.info("=" * 80)
        logger.info("Using prototype-based rank loss (cosine similarity)")
        logger.info(f"  m1: {args.rank_m1}, m2: {args.rank_m2}, beta: {args.rank_beta}, tau: {args.rank_tau}")
        logger.info(f"  K: {args.rank_K}, K': {args.rank_K_prime}")
        logger.info(f"  lambda_rank: {args.lambda_rank} (managed by trainer)")
        logger.info(f"  Rank warmup epochs: {args.rank_warmup_epochs}, ramp epochs: {args.rank_warmup_ramp_epochs}")
        logger.info("=" * 80)
    
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
        )
        logger.info(f"CosineAnnealingLR, min lr: {args.learning_rate * 0.01}")
    elif args.lr_scheduler == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True
        )
        logger.info(f"ReduceLROnPlateau, patience: {args.scheduler_patience}, factor: {args.scheduler_factor}")
    
    warmup_lr_end = args.warmup_lr if args.warmup_lr is not None else args.learning_rate
    if args.warmup_epochs > 0:
        logger.info(f"LR warmup epochs: {args.warmup_epochs}, end lr: {warmup_lr_end}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    best_acc = 0.0
    best_acc_prototype = 0.0
    best_acc_prototype_epoch = 0
    best_auroc = 0.0
    best_acc_epoch = 0
    best_auroc_epoch = 0
    best_oscr = 0.0
    best_oscr_epoch = 0
    final_test_acc = None
    final_ood_auroc = None
    
    for epoch in range(1, args.epochs + 1):
        logger.info("=" * 80)
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info("=" * 80)
        
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            warmup_lr_end = args.warmup_lr if args.warmup_lr is not None else args.learning_rate
            warmup_lr = warmup_lr_end * (epoch / args.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        if rank_loss_fn is None:
            raise ValueError("rank_loss_fn not initialized; ensure --use_rank_loss is enabled.")
        
        train_loss, train_acc, loss_dict = train_epoch_proto(
            model, train_loader, rank_loss_fn, optimizer, device, epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_epochs=args.rank_warmup_epochs,
            warmup_ramp_epochs=args.rank_warmup_ramp_epochs,
            lambda_rank=args.lambda_rank
        )
        logger.info(f"Train - loss: {train_loss:.4f}, CE: {loss_dict['ce_loss']:.4f}, "
                   f"rank_loss: {loss_dict['rank_loss']:.4f}, acc: {train_acc:.2f}%")
        
        if 'gap1' in loss_dict:
            logger.info(f"Rank - gap1: {loss_dict['gap1']:.4f}, gap2: {loss_dict['gap2']:.4f}, "
                       f"sat1: {loss_dict['sat1']:.4f}, sat2: {loss_dict['sat2']:.4f}")
        
        if epoch >= args.eval_start_epoch:
            logger.info("Computing prototypes...")
            mu_fuse = compute_fuse_prototypes(model, train_loader, args.num_classes, device)
            
            test_acc_classifier, _, _ = evaluate(model, test_loader, device)
            logger.info(f"Test - Acc (classifier): {test_acc_classifier*100:.2f}%")
            
            test_acc_prototype, _, _ = evaluate_with_prototype(
                model, test_loader, train_loader, args.num_classes, device,
                mu_fuse=mu_fuse
            )
            logger.info(f"Test - Acc (prototype): {test_acc_prototype*100:.2f}%")
            
            test_acc = test_acc_classifier
            
            ood_auroc, delta_f = evaluate_ood(
                model, test_loader, ood_loader, train_loader, args.num_classes, device, 
                delta_f=None, 
                ood_batch_size=args.ood_batch_size, use_fuse_features=True,
                percentile=args.ood_percentile, mu_fuse=mu_fuse
            )
            if delta_f is not None:
                logger.info(f"OOD - AUROC: {ood_auroc:.4f}, δ_f: {delta_f:.4f}")
            else:
                logger.info(f"OOD - AUROC: {ood_auroc:.4f}")
            
            osr_metrics = evaluate_open_set_recognition(
                model, test_loader, ood_loader, train_loader, 
                args.num_classes, device, 
                percentile=args.ood_percentile,
                ood_batch_size=args.ood_batch_size, mu_fuse=mu_fuse
            )
            logger.info(f"Open-set recognition - OSCR (prototype distance): {osr_metrics['oscr']:.4f}")
            
            # ============================
            # ============================
            if test_acc > best_acc:
                best_acc = test_acc
                best_acc_epoch = epoch
            
            if test_acc_prototype > best_acc_prototype:
                best_acc_prototype = test_acc_prototype
                best_acc_prototype_epoch = epoch
            
            if ood_auroc > best_auroc:
                best_auroc = ood_auroc
                best_auroc_epoch = epoch
            
            if osr_metrics['oscr'] > best_oscr:
                best_oscr = osr_metrics['oscr']
                best_oscr_epoch = epoch
            
            if epoch == args.epochs:
                final_test_acc = test_acc
                final_ood_auroc = ood_auroc
            
            logger.info(f"Current best - Acc (classifier): {best_acc*100:.2f}% (Epoch {best_acc_epoch}), "
                        f"Acc (prototype distance): {best_acc_prototype*100:.2f}% (Epoch {best_acc_prototype_epoch})")
            logger.info(f"Current best - AUROC (prototype distance): {best_auroc:.4f} (Epoch {best_auroc_epoch}), "
                        f"OSCR (prototype distance): {best_oscr:.4f} (Epoch {best_oscr_epoch})")
            
            # ============================
            # ============================
            if scheduler is not None and args.lr_scheduler == 'reduce_on_plateau':
                scheduler.step(test_acc)
        else:
            logger.info(f"Skip evaluation (epoch {epoch} < eval_start_epoch {args.eval_start_epoch})")
        
        if scheduler is not None and args.lr_scheduler == 'cosine':
            if args.warmup_epochs == 0 or epoch > args.warmup_epochs:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.2e}")
        
        logger.info("")
    
    # ============================
    # ============================
    model_filename = generate_rank_filename("final_model", "pth")
    final_save_path = os.path.join(args.save_dir, model_filename)
    save_dict = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_acc': best_acc,
        'best_test_acc_prototype': best_acc_prototype,
        'best_auroc': best_auroc,
        'best_oscr': best_oscr,
        'best_acc_epoch': best_acc_epoch,
        'best_acc_prototype_epoch': best_acc_prototype_epoch,
        'best_auroc_epoch': best_auroc_epoch,
        'best_oscr_epoch': best_oscr_epoch,
    }
    
    if final_test_acc is not None:
        save_dict['final_test_acc'] = final_test_acc
        save_dict['final_ood_auroc'] = final_ood_auroc
    
    torch.save(save_dict, final_save_path)
    
    logger.info("Training finished.")
    logger.info(f"Saved final model: {final_save_path}")
    logger.info(f"Best test accuracy (classifier): {best_acc*100:.2f}% (Epoch {best_acc_epoch})")
    logger.info(f"Best test accuracy (prototype distance): {best_acc_prototype*100:.2f}% (Epoch {best_acc_prototype_epoch})")
    logger.info(f"Best OOD AUROC (prototype distance): {best_auroc:.4f} (Epoch {best_auroc_epoch})")
    logger.info(f"Best OSCR (prototype distance): {best_oscr:.4f} (Epoch {best_oscr_epoch})")


if __name__ == '__main__':
    main()
