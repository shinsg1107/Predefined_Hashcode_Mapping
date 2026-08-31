"""Extract image features for PHM, for any (backbone, dataset) pair.

Companion to `finetune_encoder.py`. Loads a fine-tuned checkpoint, strips the
classifier with the generic timm call

    model.reset_classifier(0)          # -> forward() returns pooled features

so the feature width is whatever the backbone produces (768 / 1536 / 1408) with no
architecture-specific code, and writes one .pkl per split in the
`{"features": (N, D) float32, "labels": ...}` layout that PHM consumes.

Output paths are read straight out of `utils.data_utils.PATH_CONFIG`, so the files
always land exactly where training and evaluation look for them -- there is no separate
naming convention to keep in sync.

Input : raw images (see README section 2) + a checkpoint from finetune_encoder.py
Output: the .pkl files listed by PATH_CONFIG for this dataset/backbone, e.g.
        ./data/features/cifar10/{train_img768,test_img768,newdb_img768}.pkl

Examples
    python extract_features.py --backbone vit  --dataset cifar10
    python extract_features.py --backbone swin --dataset imagenet --device cuda:0
    python extract_features.py --backbone eva  --dataset nuswide --splits train test
"""
import argparse
import os
import pickle

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from finetune_encoder import BACKBONES, DATASETS, build_transforms, default_transform
from utils.data_utils import PATH_CONFIG, CIFAR10Split, ListImageDataset

# (dataset, backbone) -> the PATH_CONFIG entry that describes where the features go.
PATH_KEY = {
    ("cifar10", "vit"): "cifar-10", ("cifar10", "swin"): "cifar_swin", ("cifar10", "eva"): "cifar_eva",
    ("imagenet", "vit"): "imagenet", ("imagenet", "swin"): "imagenet_swin", ("imagenet", "eva"): "imagenet_eva",
    ("nuswide", "vit"): "nuswide", ("nuswide", "swin"): "nuswide_swin", ("nuswide", "eva"): "nuswide_eva",
}
# split name -> PATH_CONFIG field
SPLIT_FIELD = {"train": "train", "test": "query", "database": "db"}


def build_split(dataset, split, data_root, transform, multilabel):
    """Dataset object for one split, using the evaluation (non-augmenting) transform."""
    if dataset == "cifar10":
        root = data_root or "./dataset/cifar10"
        return CIFAR10Split(root, mode=split, transform=transform)

    root = data_root or f"./dataset/{dataset}"
    listdir = os.path.join(root, "setting1") if dataset == "nuswide" else root
    txt = os.path.join(listdir, f"{split}.txt")
    if not os.path.exists(txt):
        raise FileNotFoundError(
            f"image list not found: {txt}\n"
            f"Place the {dataset} lists and images under {root} (see README section 2).")
    ds = ListImageDataset(txt, transform, multilabel=multilabel, root=root)
    if ds.n_missing or ds.n_skipped_label:
        print(f"  [warn] {split}: {ds.n_listed} rows listed, {len(ds)} usable "
              f"({ds.n_missing} images missing, {ds.n_skipped_label} rows not single-label). "
              f"Missing images change the feature count and will not reproduce the reported numbers.")
    return ds


@torch.no_grad()
def extract(model, loader, device):
    feats, labels = [], []
    model.eval()
    for images, y in tqdm(loader, desc="  extracting", leave=False):
        images = images.to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            f = model(images)
        feats.append(f.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--ckpt", default=None,
                    help="fine-tuned checkpoint (default: results/image_encoder/{dataset}_{backbone}.pt)")
    ap.add_argument("--data-root", default=None, help="raw image root (default: ./dataset/{dataset})")
    ap.add_argument("--splits", nargs="+", default=["train", "test", "database"],
                    choices=["train", "test", "database"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--transform", default=None, choices=["auto", "manual"],
                    help="must match what was used for fine-tuning (default: per dataset+backbone)")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    multilabel = spec["multilabel"]
    if args.transform is None:
        args.transform = default_transform(args.dataset, args.backbone)
    if args.ckpt is None:
        args.ckpt = f"./results/image_encoder/{args.dataset}_{args.backbone}.pt"

    key = PATH_KEY[(args.dataset, args.backbone)]
    model_name, feat_dim = BACKBONES[args.backbone]
    expected_dim = PATH_CONFIG[key]["emb_size"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"=== {args.dataset} / {args.backbone} ({model_name}) ===")
    print(f"PATH_CONFIG entry '{key}', emb_size={expected_dim}, device={device}")

    if feat_dim != expected_dim:
        raise RuntimeError(
            f"BACKBONES says {args.backbone} is {feat_dim}-d but PATH_CONFIG['{key}']"
            f"['emb_size'] is {expected_dim}. Fix them together before extracting.")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f"checkpoint not found: {args.ckpt}\n"
            f"Run: python finetune_encoder.py --backbone {args.backbone} --dataset {args.dataset}")

    model = timm.create_model(model_name, pretrained=False, num_classes=spec["num_classes"])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.reset_classifier(0)          # generic across ViT / Swin / EVA
    model.to(device)

    # runtime guard: confirm the stripped model really emits `expected_dim` features
    _, val_tf = build_transforms(model, args.transform == "manual")
    with torch.no_grad():
        probe = torch.zeros(1, 3, *timm.data.resolve_model_data_config(model)["input_size"][1:])
        got = model(probe.to(device)).shape[-1]
    if got != expected_dim:
        raise RuntimeError(
            f"{model_name} emitted {got}-d features after reset_classifier(0), "
            f"but PATH_CONFIG['{key}']['emb_size'] is {expected_dim}.")
    print(f"feature dim verified: {got}")

    for split in args.splits:
        field = SPLIT_FIELD[split]
        out = PATH_CONFIG[key][field]
        print(f"\n[{split}] -> {out}")
        ds = build_split(args.dataset, split, args.data_root, val_tf, multilabel)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        feats, labels = extract(model, loader, device)
        print(f"  features {feats.shape}, labels {labels.shape}")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump({"features": feats, "labels": labels}, f)
        print(f"  saved -> {out}")

    print("\nDone. These paths are exactly what PATH_CONFIG (and therefore training "
          "and evaluation) reads, so no further moving or renaming is needed.")


if __name__ == "__main__":
    main()
