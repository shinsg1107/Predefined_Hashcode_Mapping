import torch
import numpy as np

from collections import Counter
from tqdm import tqdm
from sklearn.metrics import average_precision_score

def calculate_map_at_k(mapping_results, topk=None, device="cuda"):
    from sklearn.metrics import average_precision_score
    import torch
    import numpy as np
    from tqdm import tqdm

    query_matched_hash    = mapping_results['query_matched_hash']    # (Q, B) np.int{0,1}
    query_original_labels = mapping_results['query_original_labels'] # (Q,)
    db_matched_hash       = mapping_results['db_matched_hash']       # (N, B)
    db_original_labels    = mapping_results['db_original_labels']    # (N,)

    # convert to tensors
    db_binary_codes        = torch.from_numpy(db_matched_hash.astype(np.int8)).to(device)
    query_binary_codes     = torch.from_numpy(query_matched_hash.astype(np.int8)).to(device)
    db_original_labels_t   = torch.from_numpy(db_original_labels).to(device)
    query_original_labels_t= torch.from_numpy(query_original_labels).to(device)

    N = db_binary_codes.size(0)
    all_ap = []
    bar_format = "{l_bar}{bar}| {percentage:3.0f}%"

    for i in tqdm(range(query_binary_codes.size(0)),
                  desc=f"mAP@{topk if topk else 'all'}",
                  bar_format=bar_format):
        q_code  = query_binary_codes[i].unsqueeze(0)      # (1,B)
        q_label = query_original_labels_t[i]

        dists = torch.sum(db_binary_codes != q_code, dim=1).to(torch.float32)  # (N,)
        idx = torch.arange(N, device=device, dtype=torch.float32)
        key = dists * (N + 1) + idx / (N + 1)

        # sort
        sorted_indices = torch.argsort(key)

        if topk is not None:
            current_topk   = min(topk, N)
            sorted_indices = sorted_indices[:current_topk]

        sorted_labels = db_original_labels_t[sorted_indices]
        relevance = (sorted_labels == q_label).int().cpu().numpy()

        if np.sum(relevance) == 0:
            ap = 0.0
        else:
            # a lower key means more similar, so negate it to get a score
            ap = average_precision_score(relevance, (-key[sorted_indices]).detach().cpu().numpy())
        all_ap.append(ap)

    mean_ap = float(np.mean(all_ap)) if all_ap else 0.0
    return mean_ap

def calculate_map_at_k_multi(mapping_results, topk=None, device="cuda"):
    from sklearn.metrics import average_precision_score
    import torch
    import numpy as np
    from tqdm import tqdm

    # unpack
    q_hash = mapping_results['query_matched_hash']       # (Q, B) np.int{0,1}
    q_lbls = mapping_results['query_original_labels']    # (Q,) or (Q,C)
    d_hash = mapping_results['db_matched_hash']          # (N, B)
    d_lbls = mapping_results['db_original_labels']       # (N,) or (N,C)

    # tensors
    d_bin  = torch.from_numpy(d_hash.astype(np.int8)).to(device)
    q_bin  = torch.from_numpy(q_hash.astype(np.int8)).to(device)
    d_labT = torch.from_numpy(np.asarray(d_lbls)).to(device)
    q_labT = torch.from_numpy(np.asarray(q_lbls)).to(device)

    N = d_bin.size(0)
    all_ap = []
    bar_format = "{l_bar}{bar}| {percentage:3.0f}%"

    # multi-label vs single-label check
    multi = (d_labT.ndim == 2 and q_labT.ndim == 2)
    if multi:
        # normalize multi-hot labels to 0/1
        d_labT = (d_labT > 0).to(torch.int8)
        q_labT = (q_labT > 0).to(torch.int8)

    for i in tqdm(range(q_bin.size(0)), desc=f"mAP@{topk if topk else 'all'}", bar_format=bar_format):
        q_code  = q_bin[i].unsqueeze(0)           # (1,B)
        q_label = q_labT[i]                       # () or (C)

        # Hamming distance
        dists = torch.sum(d_bin != q_code, dim=1).to(torch.float32)  # (N,)

        # tie-breaker: equal Hamming distances are ordered by ascending index
        idx = torch.arange(N, device=device, dtype=torch.float32)
        key = dists * (N + 1) + idx / (N + 1)

        # sort
        sorted_idx = torch.argsort(key)

        if topk is not None:
            sorted_idx = sorted_idx[:min(topk, N)]

        # build the relevance vector
        if multi:
            # (K,C) & (C,) → (K,C) → any(dim=1)
            sorted_labels = d_labT[sorted_idx]                    # (K,C)
            relevance_t = (sorted_labels & q_label).any(dim=1)    # (K,)
        else:
            sorted_labels = d_labT[sorted_idx]                    # (K,)
            relevance_t = (sorted_labels == q_label)

        relevance = relevance_t.int().cpu().numpy()

        if np.sum(relevance) == 0:
            ap = 0.0
        else:
            scores = (-key[sorted_idx]).detach().cpu().numpy()    # lower is more similar → flip the sign
            ap = average_precision_score(relevance, scores)
        all_ap.append(ap)

    return float(np.mean(all_ap)) if all_ap else 0.0


def hash_distribution_concentration(mapping_results, top_n_percent=5):

    if 'db_to_hash_mapping' not in mapping_results:
        print("Error: 'db_to_hash_mapping' key not found in mapping_results.")
        return None

    db_hash_indices = np.asarray(mapping_results['db_to_hash_mapping'])
    total_images = len(db_hash_indices)
    if total_images == 0:
        print("Warning: Database mapping is empty.")
        return 0.0

    hash_counts = Counter(db_hash_indices)
    num_unique_hashes = len(hash_counts)

    if num_unique_hashes == 0:
        return 0.0

    sorted_counts = np.array([count for _, count in hash_counts.most_common()])
    k = max(1, int(num_unique_hashes * (top_n_percent / 100.0)))
    images_in_top_n = np.sum(sorted_counts[:k])
    concentration_ratio = (images_in_top_n / total_images) * 100
    print(f"\n{'='*15} HDC Results in top {top_n_percent}% {'='*15}")
    print("")
    print(f"All unique hash code: {num_unique_hashes}")
    print(f"top {top_n_percent}% hashcode: {k}")
    print(f"{top_n_percent}% hashcode mapped into {concentration_ratio:.2f}% database images")
    print(f"{'='*53}\n")
    
    return concentration_ratio


from collections import Counter
import numpy as np

def hash_distribution_concentration_topk(mapping_results, top_k=50):
    """
    Compute what percentage of the database images is covered by the
    top-K most frequently mapped hash codes.

    Args:
        mapping_results (dict): must contain a 'db_to_hash_mapping' key holding a
            1D array/list with the hash code index each DB image maps to.
        top_k (int): number of most frequent hash codes to use.

    Returns:
        ratio (float): percentage (%) of DB images covered by the top-K hash codes
        topk_hash_ids (list[int]): indices of the top-K hash codes
        topk_counts (list[int]): number of images mapped to each of those hash codes
    """
    if 'db_to_hash_mapping' not in mapping_results:
        print("Error: 'db_to_hash_mapping' key not found in mapping_results.")
        return 0.0, [], []

    db_hash_indices = np.asarray(mapping_results['db_to_hash_mapping'])
    total_images = db_hash_indices.size
    if total_images == 0:
        print("Warning: Database mapping is empty.")
        return 0.0, [], []

    # tally the occurrence count of each hash code
    hash_counts = Counter(db_hash_indices)
    num_unique_hashes = len(hash_counts)
    if num_unique_hashes == 0:
        return 0.0, [], []

    # clamp top_k into the valid range
    top_k = int(max(1, min(top_k, num_unique_hashes)))

    # take the top-K and compute the coverage ratio
    topk_items = hash_counts.most_common(top_k)  # [(hash_id, count), ...]
    images_in_topk = sum(c for _, c in topk_items)
    concentration_ratio = (images_in_topk / total_images) * 100.0

    topk_hash_ids = [h for h, _ in topk_items]
    topk_counts = [c for _, c in topk_items]

    # report
    print(f"\n{'='*15} HDC Results (top-{top_k}) {'='*15}")
    print(f"All unique hash codes: {num_unique_hashes}")
    print(f"Top-{top_k} cover: {images_in_topk}/{total_images} "
          f"({concentration_ratio:.2f}%) of DB images")
    print(f"{'='*46}\n")

    return concentration_ratio, topk_hash_ids, topk_counts
