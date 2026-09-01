"""[Deterministic] Main campaign evaluation of 18 combos: set1(codeset)·set2(test_data) mAP + HDC.
Uses perform_mapping (same as in training-time selection). metric: cifar map@All/HDC@10, imagenet map@1k/HDC@100,
nuswide map@5k(multilabel)/HDC@100. Output: results/main/main_eval.csv
"""
import os, glob, csv, re, argparse
import torch
from torch.utils.data import DataLoader
from sconf import Config
from tools import perform_mapping, perform_multilabel_mapping
from eval_codebook import build_model, encode_codebook
from utils.data_utils import data_load, DnaDataset
from utils.evaluation_metric import calculate_map_at_k, calculate_map_at_k_multi, hash_distribution_concentration_topk
from run_main_campaign import SEED

REPO = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(REPO, "results", "main")
OUT = os.path.join(MAIN, "main_eval.csv")
COMBOS = ["cifar_vit16", "cifar_vit32", "cifar_swin16", "cifar_swin32", "cifar_eva16", "cifar_eva32",
          "in_vit16", "in_vit32", "in_swin16", "in_swin32", "in_eva16", "in_eva32",
          "nus_vit16", "nus_vit32", "nus_swin16", "nus_swin32", "nus_eva16", "nus_eva32"]


def kind_of(ds):
    if ds in ("nuswide", "nuswide_swin", "nuswide_eva"):
        return "nuswide"
    if ds in ("imagenet", "imagenet_swin", "imagenet_eva"):
        return "imagenet"
    return "cifar"


def best_ckpt(root):
    pts = glob.glob(os.path.join(root, "**", "model_epoch*_map*.pt"), recursive=True)
    if not pts:
        return None, None
    def parse(p):
        m = re.search(r"model_epoch([0-9]+)_map([0-9.]+)\.pt", os.path.basename(p))
        return (int(m.group(1)), float(m.group(2))) if m else (0, -1)
    b = max(pts, key=lambda p: parse(p)[1])
    return b, parse(b)[0]


def eval_combo(job, dev):
    root = os.path.join(MAIN, job)
    ck, ep = best_ckpt(root)
    if not ck:
        print(f"[skip] {job}: no ckpt", flush=True); return None
    cfg = Config(glob.glob(os.path.join(root, "**", "config.yaml"), recursive=True)[0])
    emb_size, hash_train, hash_test, img_train, img_q, img_db = data_load(cfg)
    model = build_model(emb_size, cfg.num_label, cfg.bit, dev)
    model.load_state_dict(torch.load(ck, map_location=dev)); model.eval()
    kind = kind_of(cfg.dataset)
    topk = None if kind == "cifar" else (1000 if kind == "imagenet" else 5000)
    hdc_k = 10 if kind == "cifar" else 100

    def do(codeset):
        ds = DnaDataset(codeset, img_train, dev, cfg.num_label)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
        em, hl, binary = encode_codebook(model, loader, dev)
        if kind == "nuswide":
            mr = perform_multilabel_mapping(img_q, img_db, em, hl, binary, dev)
            mAP = float(calculate_map_at_k_multi(mr, topk=topk, device=dev))
        else:
            mr = perform_mapping(img_q, img_db, em, hl, binary, dev)
            mAP = float(calculate_map_at_k(mr, topk=topk, device=dev))
        hdc = float(hash_distribution_concentration_topk(mr, top_k=hdc_k)[0])
        return mAP, hdc

    s1_map, s1_hdc = do(hash_train)
    s2_map, s2_hdc = do(hash_test)
    r = dict(combo=job, dataset=kind, epoch=ep,
             set1_mAP=round(s1_map, 4), set1_HDC=round(s1_hdc, 2),
             set2_mAP=round(s2_map, 4), set2_HDC=round(s2_hdc, 2))
    print(f"[ok] {job}: set1 mAP={s1_map:.4f} HDC={s1_hdc:.2f} | set2 mAP={s2_map:.4f} HDC={s2_hdc:.2f} ep{ep}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    rows = []
    for job in COMBOS:
        try:
            r = eval_combo(f"{job}_P10_s{SEED}", args.device)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"[err] {job}: {type(e).__name__}: {e}", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["combo", "dataset", "epoch", "set1_mAP", "set1_HDC", "set2_mAP", "set2_HDC"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[eval_main] {len(rows)} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
