"""Fine-tune an image encoder for PHM, for any (backbone, dataset) pair.

This unifies the per-backbone / per-dataset fine-tuning scripts into one entry point.
The backbone is created with the generic timm API, so adding a backbone is just a new
entry in BACKBONES -- no architecture-specific head surgery:

    timm.create_model(model_name, pretrained=True, num_classes=N)

(The older `model.head = nn.Linear(model.head.in_features, N)` pattern is deliberately
avoided: in timm >= 1.0 the Swin head is a NormMlpClassifierHead and has no
`in_features`, so that pattern breaks for Swin.)

Input : the raw images of the chosen dataset (see README section 2).
Output: a fine-tuned checkpoint, by default
        results/image_encoder/{dataset}_{backbone}.pt
        which is what `extract_features.py` then loads.

Examples
    python finetune_encoder.py --backbone vit  --dataset cifar10
    python finetune_encoder.py --backbone swin --dataset imagenet --device cuda:0
    python finetune_encoder.py --backbone eva  --dataset nuswide  --epochs 20

Note on reproducibility: `--seed` is unset by default, which matches how the reported
checkpoints were trained. Pass `--seed` for repeatable fine-tuning runs.
"""
import argparse
import os
import random

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_utils import CIFAR10Split, ListImageDataset

# backbone key -> (timm model name, feature dim produced after reset_classifier(0))
# The feature dim must match `emb_size` in utils/data_utils.PATH_CONFIG.
BACKBONES = {
    "vit":  ("vit_base_patch16_224", 768),
    "swin": ("swin_large_patch4_window7_224.ms_in22k_ft_in1k", 1536),
    "eva":  ("eva_giant_patch14_224.clip_ft_in1k", 1408),
}

# dataset key -> defaults reproducing the per-dataset recipes these scripts unify.
DATASETS = {
    "cifar10":  dict(num_classes=10,  multilabel=False, epochs=20,  batch_size=16),
    "imagenet": dict(num_classes=100, multilabel=False, epochs=100, batch_size=32),
    "nuswide":  dict(num_classes=21,  multilabel=True,  epochs=20,  batch_size=16),
}

# Preprocessing mode per (dataset, backbone); anything not listed defaults to "auto",
# i.e. transforms derived from the backbone's own timm data config.
# The one override below was established by re-extracting features with each candidate
# transform and comparing against the released feature file: a plain 224 resize with
# ImageNet normalization reproduces it (cosine 1.0000), whereas the model's own config
# (which center-crops at crop_pct=0.9) does not (cosine 0.9287). Swin/EVA reproduce with
# "auto" (cosine 0.99999), so they are left at the default.
TRANSFORM = {
    ("cifar10", "vit"): "manual",
}


def default_transform(dataset, backbone):
    return TRANSFORM.get((dataset, backbone), "auto")

# Single-label runs use a cosine schedule; the multi-label run keeps a flat LR.
DEFAULT_LR = {False: 1e-5, True: 3e-5}
WEIGHT_DECAY = 0.01


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(model, manual):
    """Model-appropriate transforms.

    manual=False: derived from the backbone's own data config (recommended -- each
        backbone gets its expected input size and normalization).
    manual=True : plain 224 resize + ImageNet normalization, as used by the original
        NUS-WIDE script.
    """
    if manual:
        from torchvision import transforms as T
        tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return tf, tf
    cfg = timm.data.resolve_model_data_config(model)
    return (timm.data.create_transform(**cfg, is_training=True),
            timm.data.create_transform(**cfg, is_training=False))


def build_datasets(dataset, data_root, train_tf, val_tf, val_split, multilabel):
    """Return (train_dataset, val_dataset) for the chosen dataset."""
    if dataset == "cifar10":
        root = data_root or "./dataset/cifar10"
        return (CIFAR10Split(root, mode="train", transform=train_tf),
                CIFAR10Split(root, mode=val_split, transform=val_tf))

    # imagenet / nuswide are driven by image-list files
    root = data_root or f"./dataset/{dataset}"
    listdir = os.path.join(root, "setting1") if dataset == "nuswide" else root
    train_txt = os.path.join(listdir, "train.txt")
    val_txt = os.path.join(listdir, f"{val_split}.txt")
    for p in (train_txt, val_txt):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"image list not found: {p}\n"
                f"Place the {dataset} lists and images under {root} (see README section 2).")
    return (ListImageDataset(train_txt, train_tf, multilabel=multilabel, root=root),
            ListImageDataset(val_txt, val_tf, multilabel=multilabel, root=root))


@torch.no_grad()
def validate(model, loader, device, multilabel):
    model.eval()
    correct = total = 0
    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
        if multilabel:
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
        else:
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def train(model, train_loader, val_loader, device, args, multilabel):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = (None if multilabel else
                 optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            pbar.set_postfix(loss=f"{running_loss / max(seen, 1):.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        if scheduler is not None:
            scheduler.step()

        acc = validate(model, val_loader, device, multilabel)
        print(f"Epoch {epoch + 1}: validation accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  new best -> saved to {args.out}")
    print(f"Training finished. Best validation accuracy: {best_acc:.4f}")
    return best_acc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--data-root", default=None,
                    help="raw image root (default: ./dataset/{dataset})")
    ap.add_argument("--val-split", default="database", choices=["database", "test"],
                    help="split used for validation / best-checkpoint selection")
    ap.add_argument("--epochs", type=int, default=None, help="default: per-dataset")
    ap.add_argument("--batch-size", type=int, default=None, help="default: per-dataset")
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 1e-5 single-label, 3e-5 multi-label")
    ap.add_argument("--transform", default=None, choices=["auto", "manual"],
                    help="auto: derived from the backbone's data config; manual: fixed 224 "
                         "resize + ImageNet normalization (default: per dataset+backbone)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=None,
                    help="unset by default (matches the reported runs); set for repeatability")
    ap.add_argument("--out", default=None,
                    help="checkpoint path (default: results/image_encoder/{dataset}_{backbone}.pt)")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    multilabel = spec["multilabel"]
    if args.epochs is None:
        args.epochs = spec["epochs"]
    if args.batch_size is None:
        args.batch_size = spec["batch_size"]
    if args.lr is None:
        args.lr = DEFAULT_LR[multilabel]
    if args.transform is None:
        args.transform = default_transform(args.dataset, args.backbone)
    if args.out is None:
        args.out = f"./results/image_encoder/{args.dataset}_{args.backbone}.pt"

    if args.seed is not None:
        seed_everything(args.seed)
    else:
        print("[warn] --seed not set: fine-tuning is not repeatable run to run.")

    model_name, feat_dim = BACKBONES[args.backbone]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== {args.dataset} / {args.backbone} ({model_name}) ===")
    print(f"classes={spec['num_classes']} multilabel={multilabel} "
          f"epochs={args.epochs} batch={args.batch_size} lr={args.lr} device={device}")

    model = timm.create_model(model_name, pretrained=True, num_classes=spec["num_classes"])
    got = model.num_features
    if got != feat_dim:
        raise RuntimeError(
            f"{model_name} produces {got}-d features but PHM expects {feat_dim} "
            f"for backbone '{args.backbone}'. Update BACKBONES and PATH_CONFIG emb_size together.")
    print(f"feature dim: {feat_dim} (matches PATH_CONFIG emb_size)")

    train_tf, val_tf = build_transforms(model, args.transform == "manual")
    train_ds, val_ds = build_datasets(args.dataset, args.data_root, train_tf, val_tf,
                                      args.val_split, multilabel)
    print(f"train={len(train_ds)} val({args.val_split})={len(val_ds)}")

    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)

    train(model, train_loader, val_loader, device, args, multilabel)


if __name__ == "__main__":
    main()
