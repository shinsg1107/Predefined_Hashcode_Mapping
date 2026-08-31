"""
eval_codebook.py
----------------
Takes a trained hash encoder (.pt) and an arbitrary codebook CSV, and on Set1 computes
  - mAP (per-dataset metric + full breakdown)
  - HDC (Top-K concentration)
  - mapping accuracy (image label vs mapped code pseudo-label)
then saves them as JSON. (the mAP / HDC / label-accuracy trade-off of Fig.5 in the paper)

Reuses the Set1 logic of test() in main.py, split out so that only the codebook is swappable.

Usage example:
    python eval_codebook.py \
        --config config/campaign/cifar_vit16.yaml \
        --model_path results/main/cifar_vit16_P10_s2/<run>/model_epoch..._map....pt \
        --codeset_path ./codebooks/set1_cifar10_16nt.csv \
        --out ./sweep/results/cifar16_eval.json
"""
import os
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sconf import Config

from models import (DNAEncoderWithClassifier,
                    DNAEncoderWithClassifierEva,
                    DNAEncoderWithClassifierSwint)
from tools import perform_mapping, perform_multilabel_mapping
from utils.data_utils import data_load, DnaDataset
from utils.evaluation_metric import (calculate_map_at_k,
                                      calculate_map_at_k_multi,
                                      hash_distribution_concentration,
                                      hash_distribution_concentration_topk)

_MULTI = {"nuswide", "nuswide_eva", "nuswide_swin"}

# Representative metric per dataset (following Table 3 of the paper)
_METRIC_KEY = {
    "cifar-10": "map_all",
    "imagenet": "map_1k", "imagenet_swin": "map_1k", "imagenet_eva": "map_1k",
    "nuswide": "map_5k", "nuswide_eva": "map_5k", "nuswide_swin": "map_5k",
    "cifar_swin": "map_all", "cifar_eva": "map_all",
}


def build_model(embedding_size, num_label, bit, device):
    if embedding_size == 768:
        m = DNAEncoderWithClassifier(embedding_size, num_label, bit * 2)
    elif embedding_size == 1408:
        m = DNAEncoderWithClassifierEva(embedding_size, num_label, bit * 2)
    elif embedding_size == 1536:
        m = DNAEncoderWithClassifierSwint(embedding_size, num_label, bit * 2)
    else:
        raise ValueError(f"Unsupported embedding size: {embedding_size}")
    return m.to(device)


def encode_codebook(model, loader, device):
    embs, labels, raw = [], [], []
    with torch.no_grad():
        for code, label in tqdm(loader, desc="Encoding codebook"):
            raw.append(code.numpy())
            code = code.to(device, dtype=torch.float32)
            emb, _ = model(code)
            embs.append(emb.cpu().numpy())
            labels.extend(label.cpu().numpy())
    return (np.concatenate(embs, axis=0),
            np.array(labels),
            np.concatenate(raw, axis=0))


def evaluate(config, model_path, codeset_path, device=None):
    device = device or config.device
    config.codeset_path = codeset_path          # data_load override (see patch)
    num_label = config.num_label
    embedding_size, hash_train_path, _, _, img_query_path, img_db_path = data_load(config)

    is_multi = config.dataset in _MULTI

    ds = DnaDataset(hash_train_path, PATH_TRAIN_IMG(config), device, num_label)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    model = build_model(embedding_size, num_label, config.bit, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    hash_emb, hash_lab, binary_codes = encode_codebook(model, loader, device)

    if is_multi:
        mr = perform_multilabel_mapping(img_query_path, img_db_path,
                                        hash_emb, hash_lab, binary_codes, device)
        map_fn = calculate_map_at_k_multi
    else:
        mr = perform_mapping(img_query_path, img_db_path,
                             hash_emb, hash_lab, binary_codes, device)
        map_fn = calculate_map_at_k

    maps = {
        "map_all": float(map_fn(mr, topk=None, device=device)),
        "map_10k": float(map_fn(mr, topk=10000, device=device)),
        "map_5k":  float(map_fn(mr, topk=5000, device=device)),
        "map_1k":  float(map_fn(mr, topk=1000, device=device)),
    }

    # HDC (Top-K concentration, lower=better)
    hdc = {}
    for k in (10, 50, 100, 500):
        ratio, _, _ = hash_distribution_concentration_topk(mr, top_k=k)
        hdc[f"top{k}"] = float(ratio)
    for p in (1, 5):
        hdc[f"pct{p}"] = float(hash_distribution_concentration(mr, top_n_percent=p))

    metric_key = _METRIC_KEY.get(config.dataset, "map_all")

    result = {
        "dataset": config.dataset,
        "bit": int(config.bit),
        "num_label": int(num_label),
        "codeset_path": codeset_path,
        "codeset_size": int(hash_emb.shape[0]),
        "metric_key": metric_key,
        "metric_value": maps[metric_key],
        "map": maps,
        "hdc": hdc,
        "mapping_acc_query": float(mr.get("acc_q")) if mr.get("acc_q") is not None else None,
        "mapping_acc_db": float(mr.get("acc_db")) if mr.get("acc_db") is not None else None,
        "unique_codes_used_db": int(len(np.unique(mr["db_to_hash_mapping"]))),
        "model_path": model_path,
    }
    return result


# DnaDataset requires img_pkl_path (training image features), so it is injected here.
def PATH_TRAIN_IMG(config):
    from utils.data_utils import PATH_CONFIG
    return PATH_CONFIG[config.dataset]["train"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="(paper) config.yaml for this cell")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--codeset_path", required=True)
    ap.add_argument("--out", required=True, help="output path for the result JSON")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    config = Config(args.config)
    res = evaluate(config, args.model_path, args.codeset_path, device=args.device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n[eval] dataset={res['dataset']} bit={res['bit']} "
          f"size={res['codeset_size']} | {res['metric_key']}={res['metric_value']:.4f} "
          f"| HDC.top100={res['hdc']['top100']:.2f}% "
          f"| map_acc_q={res['mapping_acc_query']}")
    print(f"[eval] saved -> {args.out}")


if __name__ == "__main__":
    main()
