import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
import random
import pickle
from loss import OtLoss
from typing import Optional, Tuple, Dict, Union, List
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from sklearn.neighbors import BallTree

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # env-guard: PHM_DETERMINISTIC=1 turns on fully deterministic mode (no effect on existing runs)
    if os.environ.get("PHM_DETERMINISTIC") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as _e:
            print(f"[det] use_deterministic_algorithms warn: {_e}")
        print(f"set random seed : {seed} [DETERMINISTIC]")
    else:
        print(f"set random seed : {seed}")

def log(writer, tag, scalar_value, global_step = None, walltime = None, new_style = False, double_precision = False):
        writer.add_scalar(tag, scalar_value, global_step, walltime, new_style, double_precision)
        body = f"{tag}: {scalar_value:.6f}"
        prefix = f"Epoch {global_step + 1:03}| " if global_step is not None else ""
        logging.info(prefix + body)

def one_hot_encode(seq):
    # Bipolar (+-1) encoding, 2 values per nucleotide, so a code of length L becomes a 2L-dim
    # vector. That vector is the INPUT to the DNA encoder (input_dim = 2 * bit), not a
    # regression target: the encoder maps it to a hash embedding that the OT loss aligns with
    # image features.
    # Taking `> 0` gives the binary sign pattern A=00, T=01, C=10, G=11, and every distance in
    # this repository is a Hamming distance in that single space -- the OT pseudo-labeling cost,
    # tau_in, and tau_out alike. They are bit distances, never nucleotide Hamming distances.
    # See CODEBOOKS.md.
    mapping = {'A': [-1, -1], 'T': [-1, 1], 'C': [1, -1], 'G': [1, 1]}
    return np.array([mapping[char] for char in seq]).flatten()

def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def seq_to_array(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return np.array([mapping[b] for b in seq], dtype=np.uint8)

def _overlap_correct(true_labels, pred_labels):
    """
    true_labels, pred_labels: (N,) single-label integers or (N,C) multi-hot
    return: (N,) bool — whether each sample is correct
    """
    t = np.asarray(true_labels)
    p = np.asarray(pred_labels)

    # Single-label
    if t.ndim == 1 and p.ndim == 1:
        return (t == p)

    # Multi-label (assumed to be multi-hot)
    if t.ndim == 2 and p.ndim == 2:
        t_bin = (t > 0).astype(np.int8)
        p_bin = (p > 0).astype(np.int8)
        if t_bin.shape != p_bin.shape:
            raise ValueError(f"label shape mismatch: {t_bin.shape} vs {p_bin.shape}")
        inter = (t_bin & p_bin).sum(axis=1)
        return (inter > 0)

    # Ambiguous case where only one side is 2D is treated as an error
    raise ValueError(f"Unsupported label shapes: true={t.shape}, pred={p.shape}")


def generate_anchor_codes(
    seq_length: int = 16,
    max_gc: float = 0.5,
    max_run: int = 4,
    min_hamming: int = 2,
    fail_threshold: int = 100000,
    k_select: int = 10,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    accepted = []
    current_hamming = seq_length
    fail_count = 0

    existing_vectors = None
    tree = None

    pbar = tqdm(total=k_select, desc="Selecting anchor codes")

    while len(accepted) < k_select:
        cand = ''.join(rng.choice(bases, size=seq_length))

        # 1. GC content
        gc_ratio = (cand.count('G') + cand.count('C')) / seq_length
        if gc_ratio > max_gc:
            continue
        # 2. Homopolymer
        run, prev = 0, None
        too_long = False
        for ch in cand:
            run = run + 1 if ch == prev else 1
            prev = ch
            if run > max_run:
                too_long = True
                break
        if too_long:
            continue
        # 3. Hamming distance
        cand_vec = seq_to_array(cand)
        min_dist = float('inf')
        if existing_vectors is not None:
            dist, _ = tree.query(cand_vec.reshape(1, -1), k=1)
            min_dist = dist[0][0] * seq_length
        elif accepted:
            min_dist = min(hamming_distance(cand, a) for a in accepted)

        if min_dist < current_hamming:
            fail_count += 1
            if fail_count >= fail_threshold and current_hamming > min_hamming:
                current_hamming -= 1
                fail_count = 0
                print(f"[INFO] Hamming threshold relaxed → {current_hamming}")
            continue
        accepted.append(cand)
        existing_vectors = np.vstack([existing_vectors, cand_vec]) if existing_vectors is not None else np.array([cand_vec])
        tree = BallTree(existing_vectors, metric='hamming')
        fail_count = 0
        pbar.update(1)
    pbar.close()
    return accepted

def code_ot_mapping(onehot_code, anchor_code, device):

    onehot_code= torch.tensor(onehot_code, dtype=torch.float32).to(device)
    anchor_code = torch.tensor(anchor_code, dtype=torch.float32).to(device)
    ot = OtLoss(dist_metric='hamming', reg=1)
    _ , T = ot(onehot_code, anchor_code)
    class_labels = torch.argmax(T, dim=1).cpu().numpy()

    return T, class_labels

def perform_mapping(img_query_path, img_db_path, hash_embeddings, hash_labels, all_binary_codes, device="cuda"):

    with open(img_query_path, "rb") as f:
        query_data = pickle.load(f)
    with open(img_db_path, "rb") as f:
        db_data = pickle.load(f)

    # DB
    db_features = np.array(db_data["features"])
    db_original_labels_arr = np.array(db_data["labels"])

    # Query
    query_features = np.array(query_data["features"])
    query_original_labels_arr = np.array(query_data["labels"])

    hash_labels_arr = np.array(hash_labels)

    all_binary_codes = np.asarray(all_binary_codes)
    if all_binary_codes.dtype != np.uint8:
        all_binary_codes = (all_binary_codes > 0).astype(np.uint8)

    K = hash_embeddings.shape[0]
    assert all_binary_codes.shape[0] == K, \
        f"all_binary_codes rows ({all_binary_codes.shape[0]}) != hash_embeddings rows ({K})"
    assert hash_labels_arr.shape[0] == K, \
        f"hash_labels length ({hash_labels_arr.shape[0]}) != hash_embeddings rows ({K})"

    hash_tensor = torch.tensor(hash_embeddings, dtype=torch.float32).to(device)
    db_tensor = torch.tensor(db_features, dtype=torch.float32).to(device)
    query_tensor = torch.tensor(query_features, dtype=torch.float32).to(device)

    hash_tensor_norm = F.normalize(hash_tensor, p=2, dim=1)
    db_tensor_norm = F.normalize(db_tensor, p=2, dim=1)
    query_tensor_norm = F.normalize(query_tensor, p=2, dim=1)

    similarity_db = torch.matmul(hash_tensor_norm, db_tensor_norm.T)
    similarity_query = torch.matmul(hash_tensor_norm, query_tensor_norm.T)

    db_to_hash_mapping_indices = torch.argmax(similarity_db, dim=0).cpu().numpy() if similarity_db.numel() > 0 else np.array([], dtype=int)
    query_to_hash_mapping_indices = torch.argmax(similarity_query, dim=0).cpu().numpy() if similarity_query.numel() > 0 else np.array([], dtype=int)


    db_matched_hash_embeddings = hash_embeddings[db_to_hash_mapping_indices] if len(db_to_hash_mapping_indices) > 0 else np.array([])
    query_matched_hash_embeddings = hash_embeddings[query_to_hash_mapping_indices] if len(query_to_hash_mapping_indices) > 0 else np.array([])
    
    db_matched_hash_binaries = all_binary_codes[db_to_hash_mapping_indices] if len(db_to_hash_mapping_indices) > 0 else np.array([])
    query_matched_hash_binaries = all_binary_codes[query_to_hash_mapping_indices] if len(query_to_hash_mapping_indices) > 0 else np.array([])

    db_matched_hash_pseudo_labels = hash_labels_arr[db_to_hash_mapping_indices] if len(db_to_hash_mapping_indices) > 0 else np.array([])
    query_matched_hash_pseudo_labels = hash_labels_arr[query_to_hash_mapping_indices] if len(query_to_hash_mapping_indices) > 0 else np.array([])


    if len(query_original_labels_arr) > 0 and len(query_matched_hash_pseudo_labels) == len(query_original_labels_arr):
        correct_q = (query_original_labels_arr == query_matched_hash_pseudo_labels)
        accuracy_q = np.mean(correct_q) * 100
        print(f"Query acc (vs hash label): {accuracy_q:.2f}%")

    if len(db_original_labels_arr) > 0 and len(db_matched_hash_pseudo_labels) == len(db_original_labels_arr):
        correct_db = (db_original_labels_arr == db_matched_hash_pseudo_labels)
        accuracy_db = np.mean(correct_db) * 100
        print(f"DB acc (vs hash label): {accuracy_db:.2f}%")

    mapping_results = {
        'query_matched_hash': query_matched_hash_binaries,
        'query_original_labels': query_original_labels_arr,
        'db_matched_hash': db_matched_hash_binaries,
        'db_original_labels': db_original_labels_arr,
        'db_to_hash_mapping': db_to_hash_mapping_indices,
        'query_to_hash_mapping': query_to_hash_mapping_indices,
        'query_matched_feature': query_matched_hash_embeddings,
        'db_matched_feature': db_matched_hash_embeddings,
        'acc_db' : accuracy_db,
        'acc_q' : accuracy_q,
    }

    return mapping_results

def perform_multilabel_mapping(img_query_path, img_db_path, hash_embeddings, hash_labels, all_binary_codes, device="cuda"):

    with open(img_query_path, "rb") as f:
        query_data = pickle.load(f)
    with open(img_db_path, "rb") as f:
        db_data = pickle.load(f)

    # DB
    db_features = np.array(db_data["features"])
    db_original_labels_arr = np.array(db_data["labels"])

    # Query
    query_features = np.array(query_data["features"])
    query_original_labels_arr = np.array(query_data["labels"])

    hash_labels_arr = np.array(hash_labels)

    all_binary_codes = np.asarray(all_binary_codes)
    if all_binary_codes.dtype != np.uint8:
        all_binary_codes = (all_binary_codes > 0).astype(np.uint8)

    K = hash_embeddings.shape[0]
    assert all_binary_codes.shape[0] == K, \
        f"all_binary_codes rows ({all_binary_codes.shape[0]}) != hash_embeddings rows ({K})"
    assert hash_labels_arr.shape[0] == K, \
        f"hash_labels length ({hash_labels_arr.shape[0]}) != hash_embeddings rows ({K})"

    hash_tensor = torch.tensor(hash_embeddings, dtype=torch.float32).to(device)
    db_tensor = torch.tensor(db_features, dtype=torch.float32).to(device)
    query_tensor = torch.tensor(query_features, dtype=torch.float32).to(device)

    hash_tensor_norm = F.normalize(hash_tensor, p=2, dim=1)
    db_tensor_norm = F.normalize(db_tensor, p=2, dim=1)
    query_tensor_norm = F.normalize(query_tensor, p=2, dim=1)

    similarity_db = torch.matmul(hash_tensor_norm, db_tensor_norm.T)
    similarity_query = torch.matmul(hash_tensor_norm, query_tensor_norm.T)

    db_to_hash_mapping_indices = torch.argmax(similarity_db, dim=0).cpu().numpy() if similarity_db.numel() > 0 else np.array([], dtype=int)
    query_to_hash_mapping_indices = torch.argmax(similarity_query, dim=0).cpu().numpy() if similarity_query.numel() > 0 else np.array([], dtype=int)


    db_matched_hash_embeddings = hash_embeddings[db_to_hash_mapping_indices] if len(db_to_hash_mapping_indices) > 0 else np.array([])
    query_matched_hash_embeddings = hash_embeddings[query_to_hash_mapping_indices] if len(query_to_hash_mapping_indices) > 0 else np.array([])
    
    db_matched_hash_binaries = all_binary_codes[db_to_hash_mapping_indices] if len(db_to_hash_mapping_indices) > 0 else np.array([])
    query_matched_hash_binaries = all_binary_codes[query_to_hash_mapping_indices] if len(query_to_hash_mapping_indices) > 0 else np.array([])

    db_matched_hash_pseudo_labels = hash_labels_arr[db_to_hash_mapping_indices] if len(db_to_hash_mapping_indices) > 0 else np.array([])
    query_matched_hash_pseudo_labels = hash_labels_arr[query_to_hash_mapping_indices] if len(query_to_hash_mapping_indices) > 0 else np.array([])

    try:
        if len(query_original_labels_arr) > 0 and len(query_matched_hash_pseudo_labels) == len(query_original_labels_arr):
            correct_q = _overlap_correct(query_original_labels_arr, query_matched_hash_pseudo_labels)
            accuracy_q = float(np.mean(correct_q) * 100.0)
            print(f"Query acc (label overlap): {accuracy_q:.2f}%")

        if len(db_original_labels_arr) > 0 and len(db_matched_hash_pseudo_labels) == len(db_original_labels_arr):
            correct_db = _overlap_correct(db_original_labels_arr, db_matched_hash_pseudo_labels)
            accuracy_db = float(np.mean(correct_db) * 100.0)
            print(f"DB acc (label overlap): {accuracy_db:.2f}%")
    except Exception as e:
        print(f"[warn] label-accuracy print skipped: {e}")

    mapping_results = {
        'query_matched_hash': query_matched_hash_binaries,
        'query_original_labels': query_original_labels_arr,
        'db_matched_hash': db_matched_hash_binaries,
        'db_original_labels': db_original_labels_arr,
        'db_to_hash_mapping': db_to_hash_mapping_indices,
        'query_to_hash_mapping': query_to_hash_mapping_indices,
        'query_matched_feature': query_matched_hash_embeddings,
        'db_matched_feature': db_matched_hash_embeddings
    }

    return mapping_results

def probabilistic_topk_sampler_simple(sim_mat: torch.Tensor,
                                      k: int = 3,
                                      seed: int = 0,
                                      mode: str = "softmax") -> np.ndarray:
    if sim_mat.numel() == 0:
        return np.array([], dtype=np.int64)

    # Do everything on CPU (safe)
    M = sim_mat.detach().cpu().to(torch.float32).contiguous()  # [K, N]
    M[~torch.isfinite(M)] = -1e9

    K, N = M.shape
    k = int(max(1, min(k, K)))

    # Per-column top-k scores/indices
    top_scores, top_idx = torch.topk(M, k=k, dim=0)  # [k, N]

    # Build probabilities
    if mode == "softmax":
        # Normalize to prevent overflow, then softmax
        z = top_scores - top_scores.max(dim=0, keepdim=True).values
        probs = torch.softmax(z, dim=0)  # [k, N]
    elif mode == "proportional":
        # Handle negatives/zeros: subtract per-column min to make non-negative; uniform if the sum is 0
        w = top_scores - top_scores.min(dim=0, keepdim=True).values
        w = torch.clamp(w, min=0)
        s = w.sum(dim=0, keepdim=True)
        probs = torch.where(s > 0, w / s, torch.full_like(w, 1.0 / k))
    else:
        raise ValueError("mode must be 'softmax' or 'proportional'")

    # Draw 1 sample per column (multinomial)
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))
    # multinomial samples row-wise distributions, so transpose
    choice_rel = torch.multinomial(probs.t().contiguous(), num_samples=1, replacement=True, generator=g).squeeze(1)  # [N]
    # Map relative → absolute indices
    chosen = torch.gather(top_idx, 0, choice_rel.unsqueeze(0)).squeeze(0)  # [N]

    return chosen.numpy().astype(np.int64)


def perform_soft_mapping(img_query_path, img_db_path, hash_embeddings, hash_labels, all_binary_codes, device="cuda"):

    with open(img_query_path, "rb") as f:
        query_data = pickle.load(f)
    with open(img_db_path, "rb") as f:
        db_data = pickle.load(f)

    # DB
    db_features = np.array(db_data["features"])
    db_original_labels_arr = np.array(db_data["labels"])

    # Query
    query_features = np.array(query_data["features"])
    query_original_labels_arr = np.array(query_data["labels"])

    hash_labels_arr = np.array(hash_labels)

    all_binary_codes = np.asarray(all_binary_codes)
    if all_binary_codes.dtype != np.uint8:
        all_binary_codes = (all_binary_codes > 0).astype(np.uint8)

    K = hash_embeddings.shape[0]
    assert all_binary_codes.shape[0] == K, \
        f"all_binary_codes rows ({all_binary_codes.shape[0]}) != hash_embeddings rows ({K})"
    assert hash_labels_arr.shape[0] == K, \
        f"hash_labels length ({hash_labels_arr.shape[0]}) != hash_embeddings rows ({K})"

    hash_tensor = torch.tensor(hash_embeddings, dtype=torch.float32).to(device)
    db_tensor = torch.tensor(db_features, dtype=torch.float32).to(device)
    query_tensor = torch.tensor(query_features, dtype=torch.float32).to(device)

    hash_tensor_norm = F.normalize(hash_tensor, p=2, dim=1)
    db_tensor_norm = F.normalize(db_tensor, p=2, dim=1)
    query_tensor_norm = F.normalize(query_tensor, p=2, dim=1)

    similarity_db = torch.matmul(hash_tensor_norm, db_tensor_norm.T)
    similarity_query = torch.matmul(hash_tensor_norm, query_tensor_norm.T)

    db_to_hash_mapping_indices = probabilistic_topk_sampler_simple(similarity_db, k=3, seed=42, mode="softmax")
    query_to_hash_mapping_indices = probabilistic_topk_sampler_simple(similarity_query, k=3, seed=42, mode="softmax")



    db_matched_hash_embeddings = hash_embeddings[db_to_hash_mapping_indices] if len(db_to_hash_mapping_indices) > 0 else np.array([])
    query_matched_hash_embeddings = hash_embeddings[query_to_hash_mapping_indices] if len(query_to_hash_mapping_indices) > 0 else np.array([])
    
    db_matched_hash_binaries = all_binary_codes[db_to_hash_mapping_indices] if len(db_to_hash_mapping_indices) > 0 else np.array([])
    query_matched_hash_binaries = all_binary_codes[query_to_hash_mapping_indices] if len(query_to_hash_mapping_indices) > 0 else np.array([])

    db_matched_hash_pseudo_labels = hash_labels_arr[db_to_hash_mapping_indices] if len(db_to_hash_mapping_indices) > 0 else np.array([])
    query_matched_hash_pseudo_labels = hash_labels_arr[query_to_hash_mapping_indices] if len(query_to_hash_mapping_indices) > 0 else np.array([])


    if len(query_original_labels_arr) > 0 and len(query_matched_hash_pseudo_labels) == len(query_original_labels_arr):
        correct_q = (query_original_labels_arr == query_matched_hash_pseudo_labels)
        accuracy_q = np.mean(correct_q) * 100
        print(f"Query acc (vs hash label): {accuracy_q:.2f}%")

    if len(db_original_labels_arr) > 0 and len(db_matched_hash_pseudo_labels) == len(db_original_labels_arr):
        correct_db = (db_original_labels_arr == db_matched_hash_pseudo_labels)
        accuracy_db = np.mean(correct_db) * 100
        print(f"DB acc (vs hash label): {accuracy_db:.2f}%")

    mapping_results = {
        'query_matched_hash': query_matched_hash_binaries,
        'query_original_labels': query_original_labels_arr,
        'db_matched_hash': db_matched_hash_binaries,
        'db_original_labels': db_original_labels_arr,
        'db_to_hash_mapping': db_to_hash_mapping_indices,
        'query_to_hash_mapping': query_to_hash_mapping_indices,
        'query_matched_feature': query_matched_hash_embeddings,
        'db_matched_feature': db_matched_hash_embeddings
    }

    return mapping_results

def build_inputs_for_mapping(dna_train_loader, dna_model, device, num_classes, dataset_name):
    dna_model.eval()
    emb_chunks = []
    code_chunks = []
    label_chunks = []

    _MULTI = {'nuswide', 'nuswide_eva', 'nuswide_swin'}
    is_multi = dataset_name in _MULTI

    with torch.no_grad():
        for code, label in tqdm(dna_train_loader, desc="Encoding Hash"):
            # raw binary codes (assumed to be CPU tensors)
            code_chunks.append(code.numpy())

            # extract hash embeddings
            code = code.to(device, dtype=torch.float32)
            emb, _ = dna_model(code)
            emb_chunks.append(emb.detach().cpu().numpy())

            # organize labels
            if is_multi:
                # case A) already multi-hot (B, C)
                if label.ndim == 2:
                    mh = (label > 0).to(torch.int8)
                # case B) single/multiple integer indices (B,) → one-hot
                elif label.ndim == 1:
                    mh = F.one_hot(label.to(torch.int64), num_classes=num_classes).to(torch.int8)
                else:
                    raise ValueError(f"Unsupported label shape for multi-label: {tuple(label.shape)}")
                label_chunks.append(mh.cpu().numpy())
            else:
                # single-label (B,) integers
                label_chunks.append(label.cpu().numpy().astype(np.int64))

    hash_embeddings = np.concatenate(emb_chunks, axis=0)          # (K, D)
    all_binary_codes = np.concatenate(code_chunks, axis=0)        # (K, B)
    anchor_labels   = np.concatenate(label_chunks, axis=0)        # (K,) or (K, C)

    return hash_embeddings, anchor_labels, all_binary_codes

def CalcHammingDist(B1, B2):
    return np.sum(np.not_equal(B1, B2).astype(np.int8), axis=1)

def calculate_pr_curve(mapping_result,
                               interval=100,
                               topk=None,
                               multilabel=False):

    q_labels = np.asarray(mapping_result['query_original_labels'])
    d_labels = np.asarray(mapping_result['db_original_labels'])
    Q = int(len(q_labels))
    d_codes = np.asarray(mapping_result['db_matched_hash']).astype(np.uint8)
    q_codes = np.asarray(mapping_result['query_matched_hash']).astype(np.uint8)
    N = d_codes.shape[0]
    if topk is None:
        topk = N
    topk = min(topk, N)

    # Diagnostic counters
    zero_pos = 0
    pos_counts = []
    tie_cols = 0   # number of queries that had ties within the top-k

    # Average PR over valid queries only (positives≥1)
    prec_list = []
    rec_list  = []

    for i in tqdm(range(Q), desc="PR (aligned w/ mAP)"):
        # Ground-truth vector
        if multilabel:
            # When q_labels[i] and d_labels are in {0,1}^C, use label intersection
            gnd = (d_labels @ q_labels[i]) > 0
        else:
            gnd = (d_labels == q_labels[i])
        gnd = gnd.astype(np.int32)
        n_pos = int(gnd.sum())
        if n_pos == 0:
            zero_pos += 1
            continue
        pos_counts.append(n_pos)

        # Hamming distance + index tie-break (same policy as mAP)
        dist = np.sum(d_codes != q_codes[i], axis=1).astype(np.int32)  # (N,)
        order = np.lexsort((np.arange(N, dtype=np.int64), dist))       # dist↑, idx↑
        order = order[:topk]
        g = gnd[order]

        # Whether there were ties within the top-k
        if np.any(np.diff(dist[order]) == 0):
            tie_cols += 1

        ctp = np.cumsum(g)                         # cumulative TP
        k_vals = np.arange(1, topk+1)
        prec = ctp / k_vals
        rec  = ctp / n_pos

        prec_list.append(prec)
        rec_list.append(rec)

    if len(prec_list) == 0:
        print("[WARN] no valid(positives≥1) query found.")
        return [], [], []

    # Macro average
    P = np.mean(np.stack(prec_list, axis=0), axis=0)  # (topk,)
    R = np.mean(np.stack(rec_list,  axis=0), axis=0)  # (topk,)

    # interval sampling
    idxs = np.arange(0, topk, interval)
    if len(idxs) == 0 or idxs[-1] != topk-1:
        idxs = np.append(idxs, topk-1)

    # Diagnostic log
    print("\n[PR diagnostics]")
    print(f"- total queries: {Q}")
    print(f"- valid queries (positives≥1): {len(prec_list)}")
    print(f"- zero-positive queries (excluded): {zero_pos}")
    if pos_counts:
        pos_arr = np.array(pos_counts)
        print(f"- positives per valid query: mean={pos_arr.mean():.1f}, "
              f"median={np.median(pos_arr):.0f}, min={pos_arr.min()}, max={pos_arr.max()}")
    print(f"- queries w/ ties in top-{topk}: {tie_cols} / {len(prec_list)}")

    return P[idxs].tolist(), R[idxs].tolist(), (idxs+1).tolist()


def calculate_pr_curve_multi(mapping_result, interval=100, topk=None, multilabel=False):
    """
    Computes P@k / R@k curve data from the mapping results.
    """
    q_labels = np.asarray(mapping_result['query_original_labels'])
    d_labels = np.asarray(mapping_result['db_original_labels'])
    q_codes = np.asarray(mapping_result['query_matched_hash']).astype(np.uint8)
    d_codes = np.asarray(mapping_result['db_matched_hash']).astype(np.uint8)

    Q = q_codes.shape[0]
    N = d_codes.shape[0]
    
    if topk is None:
        topk = N
    topk = min(topk, N)

    prec_list = []
    rec_list  = []
    valid_queries = 0

    for i in tqdm(range(Q), desc="Calculating P-R Curve"):
        # Build the ground truth vector
        if multilabel:
            gnd = (d_labels @ q_labels[i] > 0).astype(np.float32)
        else:
            gnd = (d_labels == q_labels[i]).astype(np.float32)
        
        n_pos = int(np.sum(gnd))
        if n_pos == 0:
            continue
        valid_queries +=1

        # Compute Hamming distances and sort
        dist = np.sum(d_codes != q_codes[i], axis=1).astype(np.int32)
        order = np.argsort(dist)
        gnd_sorted = gnd[order][:topk]

        # Compute precision and recall
        ctp = np.cumsum(gnd_sorted)
        k_vals = np.arange(1, topk + 1)
        prec = ctp / k_vals
        rec  = ctp / n_pos

        prec_list.append(prec)
        rec_list.append(rec)

    if valid_queries == 0:
        print("[WARN] No valid queries with at least one positive sample found.")
        return [], [], []

    # Macro average over all valid queries
    P = np.mean(np.stack(prec_list, axis=0), axis=0)
    R = np.mean(np.stack(rec_list, axis=0), axis=0)

    # Interval sampling
    idxs = np.arange(0, topk, interval)
    if not idxs.size or idxs[-1] != topk - 1:
        idxs = np.append(idxs, topk - 1)

    return P[idxs].tolist(), R[idxs].tolist(), (idxs + 1).tolist()


def calculate_pr_curve_old(mapping_result, interval=100):
    """
    Function that computes Precision-Recall curve data. (CPU-based NumPy operations)
    """
    query_labels = mapping_result['query_original_labels']
    db_labels = mapping_result['db_original_labels']
    
    query_binary_codes = (mapping_result['query_matched_hash'] > 0).astype(np.int8)
    db_binary_codes = (mapping_result['db_matched_hash'] > 0).astype(np.int8)

    num_query, num_db = query_binary_codes.shape[0], db_binary_codes.shape[0]
    
    prec_matrix = np.zeros((num_query, num_db))
    recall_matrix = np.zeros((num_query, num_db))

    for i in tqdm(range(num_query), desc="Calculating P-R Curve (CPU)"):
        
        gnd = (query_labels[i] == db_labels).astype(np.float32)

        if np.sum(gnd) == 0:
            continue

        hamm_dist = CalcHammingDist(query_binary_codes[i, :], db_binary_codes)
        sorted_indices = np.argsort(hamm_dist)
        gnd_sorted = gnd[sorted_indices]

        cumulative_tps = np.cumsum(gnd_sorted)
        total_relevant_items = np.sum(gnd)
        
        k_values = np.arange(1, num_db + 1)
        
        prec_matrix[i, :] = cumulative_tps / k_values
        recall_matrix[i, :] = cumulative_tps / total_relevant_items

    avg_precision = np.mean(prec_matrix, axis=0)
    avg_recall = np.mean(recall_matrix, axis=0)

    sample_indices = np.arange(0, num_db, interval)
    
    P_list = avg_precision[sample_indices].tolist()
    R_list = avg_recall[sample_indices].tolist()
    index_list = (sample_indices + 1).tolist()

    print(f"CPU-based P-R Curve data generation done. {len(index_list)} points.")
    
    return P_list, R_list, index_list