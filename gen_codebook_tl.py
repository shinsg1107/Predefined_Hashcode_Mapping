"""Codebook generation: biologically-valid pool -> OT pseudo-labels -> tau_in clique -> tau_out margin.

Every distance in this pipeline is counted in ONE space: the sign pattern of
`tools.one_hot_encode`, i.e. A=00, T=01, C=10, G=11, two bits per nucleotide. Both thresholds
live there, so a tau value means the same thing at every stage. See CODEBOOKS.md.

Stages
  1. OT pseudo-labeling  each pool code is assigned to the class whose anchor it transports to
  2. tau_in max-clique   per class, the largest set whose codes are all mutually within tau_in
                         (bitset greedy, fixed restarts and seed -> reproducible)
  3. tau_out margin      drop any code with more than `m` other-class neighbours within tau_out,
                         comparing against the codebook itself (deterministic, no sampling)

Settings behind the shipped codebooks (see CODEBOOKS.md for the full table):
    16nt : tau_in=12, tau_out=3,  m=0
    32nt : tau_in=34, tau_out=16, m=0   (CIFAR-10 32nt uses tau_in=32)

Example -- regenerate the shipped CIFAR-10 16nt codebook:

    python gen_codebook_tl.py \
        --pool dataset/codeset/pool_16nt_100k.csv \
        --anchors dataset/codeset/anchor_codes_10.csv \
        --num-label 10 --tau-in 12 --tau-out 3 --max-bad 0 \
        --device cuda:0 --out codebooks/my_cifar10_16nt.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from clique import build_adjacency, greedy_clique, verify_clique
from tools import one_hot_encode, code_ot_mapping, set_seed


def sign_bits(codes):
    """Codes -> 0/1 matrix in the sign space of tools.one_hot_encode (A=00, T=01, C=10, G=11)."""
    return (np.array([one_hot_encode(c) for c in codes], dtype=np.float32) > 0).astype(np.uint8)


def pseudo_label(codes, anchor_codes, device):
    feats = np.array([one_hot_encode(c) for c in codes], dtype=np.float32)
    afeats = np.array([one_hot_encode(c) for c in anchor_codes], dtype=np.float32)
    _, labels = code_ot_mapping(feats, afeats, device)
    return np.asarray(labels)


def clique_stage(df, X, labels, num_label, tau_in, restarts, seed):
    """Per-class tau_in max-clique; the union is the pre-margin ('expanded') codebook."""
    parts = []
    for lab in range(num_label):
        idx = np.where(labels == lab)[0]
        if idx.size < 2:
            print(f"[gen] class {lab}: {idx.size} codes, skipped")
            continue
        A = build_adjacency(X[idx], tau_in)
        cl = greedy_clique(A, restarts=restarts, seed=seed)
        if not verify_clique(X[idx], cl, tau_in):
            raise RuntimeError(f"class {lab}: greedy result is not a clique at tau_in={tau_in}")
        sub = df.iloc[idx[np.asarray(cl, dtype=int)]].copy()
        sub["label"] = lab
        parts.append(sub)
        print(f"[gen] class {lab}: pool={idx.size} clique={len(cl)}", flush=True)
    if not parts:
        raise RuntimeError("no class produced a clique; check tau_in")
    return pd.concat(parts)[["code", "label"]].reset_index(drop=True)


def margin_stage(df, tau_out, max_bad):
    """Drop codes with more than `max_bad` other-class neighbours within tau_out.

    Compared against the codebook itself, so the result does not depend on any sampling.
    """
    X = sign_bits(df["code"].values)
    L = df["label"].values
    keep = []
    for i in range(len(X)):
        other = X[L != L[i]]
        if other.shape[0] == 0:
            keep.append(i)
            continue
        bad = int(np.count_nonzero((other != X[i]).sum(axis=1) <= tau_out))
        if bad <= max_bad:
            keep.append(i)
    print(f"[gen] margin(tau_out={tau_out}, m={max_bad}): {len(X)} -> {len(keep)}")
    return df.iloc[keep].reset_index(drop=True)


def report(df):
    X = sign_bits(df["code"].values)
    L = df["label"].values
    intra = 0
    for lab in np.unique(L):
        Y = X[L == lab]
        if len(Y) > 1:
            intra = max(intra, int((Y[:, None] != Y[None]).sum(-1).max()))
    cross = min(int(np.min((X[L != L[i]] != X[i]).sum(1))) for i in range(len(X)))
    cnt = np.bincount(L, minlength=int(L.max()) + 1)
    print(f"[gen] {len(df)} codes, {len(np.unique(L))} classes, per-class {cnt.min()}-{cnt.max()}")
    print(f"[gen] max intra-class distance = {intra}, min inter-class distance = {cross}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="biologically-valid pool CSV with a 'code' column")
    ap.add_argument("--anchors", required=True, help="anchor code CSV (one per class)")
    ap.add_argument("--num-label", type=int, required=True)
    ap.add_argument("--tau-in", type=int, required=True, help="clique threshold (sign-space bits)")
    ap.add_argument("--tau-out", type=int, required=True, help="margin radius (same space)")
    ap.add_argument("--max-bad", type=int, default=0,
                    help="tolerated other-class neighbours within tau_out (shipped codebooks: 0)")
    ap.add_argument("--restarts", type=int, default=16, help="greedy clique restarts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-expanded", default=None,
                    help="optional path for the pre-margin clique union")
    args = ap.parse_args()

    df = pd.read_csv(args.pool)
    if "code" not in df.columns:
        raise ValueError(f"pool CSV has no 'code' column: {list(df.columns)}")
    anchors = pd.read_csv(args.anchors)["code"].values
    if len(anchors) != args.num_label:
        print(f"[gen][warn] {len(anchors)} anchors for num_label={args.num_label}")
    print(f"[gen] pool={len(df)} anchors={len(anchors)} tau_in={args.tau_in} "
          f"tau_out={args.tau_out} m={args.max_bad}")

    set_seed(args.seed)
    labels = pseudo_label(df["code"].values, anchors, args.device)
    X = sign_bits(df["code"].values)

    expanded = clique_stage(df, X, labels, args.num_label, args.tau_in, args.restarts, args.seed)
    print(f"[gen] clique union = {len(expanded)} codes")
    if args.save_expanded:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_expanded)), exist_ok=True)
        expanded.to_csv(args.save_expanded, index=False)

    out = margin_stage(expanded, args.tau_out, args.max_bad)
    report(out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[gen] wrote {args.out}")


if __name__ == "__main__":
    main()
