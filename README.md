# Breaking-Background_Bias

This folder contains the implementation of a **foreground–background fusion model** based on **DINOv3** for **breaking background bias** and improving **Open-Set Recognition (OSR)** performance on ImageNet.

The method explicitly models foreground and background features, fuses them with cross-attention, and uses a ranking loss on foreground–background pairs to encourage robust, background-invariant representations.

## Overview

- **Backbone**: DINOv3 ViT-S/16  
  - First 8 transformer blocks are frozen, last 4 blocks are trainable.
- **Inputs**:  
  - Foreground image `x_fg` and background image `x_bg` for each sample (obtained from a single ImageNet image via foreground–background separation).
- **Fusion**:  
  - Multi-layer cross-attention transformer to fuse foreground and background patch tokens.  
  - Outputs: fused feature, foreground feature, background feature.
- **Classifier**:  
  - MLP projection + cosine classifier.
- **Losses**:
  - Standard cross-entropy (CE) for classification.
  - **Three-level ranking loss `RankLoss`** (see `loss.py`), defined on log-odds:
    - True pair `(fg_i, bg_i)` should have the highest log-odds.
    - Fixed foreground + mismatched background `(fg_i, bg_j)` should be lower.
    - Mismatched foreground + fixed background `(fg_k, bg_i)` should be the lowest.
- **Evaluation**:
  - Accuracy (ACC), AUROC, OSCR, and open-set recognition metrics (see `evaluator.py`).

## Code Structure

- `train.py`  
  Main training script.
  - Parses command-line arguments (dataset type, learning rate, epochs, loss hyperparameters `m1/m2/K/K'`, device, etc.).
  - Builds `FusionClassifier`, optimizer, scheduler and loss.
  - Training loop with periodic evaluation and model checkpointing.

- `model.py`  
  - `DINOv3Backbone`: loads and partially fine-tunes DINOv3 (last 4 blocks trainable).  
  - `FusionClassifier`: foreground/background feature extraction, multi-layer cross-attention fusion, projection head and cosine classifier.

- `dataset.py`  
  - `ImageNetAnimalFusionDataset`: loads foreground and background images for ImageNet animal classes.  
    - Uses ID / OOD class list files from `imagenet_animial_split`.  
    - Supports cached index files for fast loading.

- `loss.py`  
  - `compute_log_odds`: one-vs-rest log-odds for multiclass logits.  
  - `RankLoss`: three-level ranking loss on true/mismatched foreground–background pairs.

- `loss_proto.py`, `trainer_proto.py`, `evaluator.py`  
  - Prototype-based training and evaluation utilities (e.g., computing fused prototypes, open-set metrics).

- `utils.py`  
  - Utility functions such as logging (`setup_logging`) and seeding (`set_seed`).

- `tool/visualize_max_activation.py`  
  - Visualizes KDE distributions of **maximum activations** for **one model** (fused / foreground / background), using ID vs OOD samples.

- `tool/visualize_max_activation_all.py`  
  - Unified visualization for:
    - **Baseline** model (backbone features, classifier-input features),
    - **λ=0 (CE-only)** fusion model,
    - **Ours** fusion model.
  - Computes and caches features once (NumPy `.npz`), and reuses them for repeated plotting.
  - Supports GPU acceleration when available.

## Data and Class Splits

This project assumes:

- Preprocessed ImageNet-1K (ILSVRC2012) data under  
  `dataset/Processed_Data/ImageNet` (not included here).
- Class-split files provided in `imagenet_animial_split/` (see its own `README.md`):
  - `imagenet_animal_id_classes.txt`, `imagenet_animal_ood_classes.txt`
  - `imagenet_object_id_classes.txt`, `imagenet_object_ood_classes.txt`
  - `imagenet_other_id_classes.txt`, `imagenet_other_ood_classes.txt`

`ImageNetAnimalFusionDataset` reads these files to construct ID / OOD splits and foreground/background paths.

## Training and Evaluation Examples

```bash
# 1. Train the Ours fusion model (example hyperparameters)
python train.py \
  --dataset animal \
  --imagenet_root /path/to/Processed_Data/ImageNet \
  --id_class_list /path/to/imagenet_animal_id_classes.txt \
  --ood_class_list /path/to/imagenet_animal_ood_classes.txt \
  --pretrained_path /path/to/dinov3_vits16_pretrain.pth \
  --epochs 50 \
  --batch_size 48 \
  --device cuda

# 2. Visualize max activation distribution for a single fusion model
python tool/visualize_max_activation.py \
  --model_path checkpoints/final_model_....pth \
  --dataset animal \
  --device cuda

# 3. Joint visualization for Baseline / λ=0 / Ours
python tool/visualize_max_activation_all.py --device cuda
```

(Adjust paths and hyperparameters according to your environment.)

## Environment

- Python 3.8+  
- PyTorch (with CUDA recommended)  
- torchvision, numpy, scipy, matplotlib, tqdm, Pillow, etc.  
- DINOv3 codebase and pretrained weights (path configurable via `--pretrained_path` and in `model.py`).

## License and Citation

- Please follow the official **ImageNet** terms of use for the data.  
- If you use this codebase or the provided class splits in your work, please cite this repository or the corresponding paper.

