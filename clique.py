"""Bitset greedy maximum-clique, used by the tau_in stage of codebook generation.

Codes are compared in the sign space of `tools.one_hot_encode` (see CODEBOOKS.md): two bits
per nucleotide, so a 16nt code is a 32-bit vector and a 32nt code is a 64-bit vector. The
adjacency matrix is stored packed (one bit per pair), which keeps even a 10k-node label group
to a few MB.

A greedy expansion with random restarts is used rather than an exact solver: the tau_in graphs
are 35-45% dense with thousands of nodes per label, where exact max-clique is intractable and
the usual complement-graph approximations blow up in memory. Restart count and seed are fixed
so the result is reproducible.
"""
import numpy as np

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def build_adjacency(bits_mat, thr, chunk=512):
    """(n, B) 0/1 matrix -> packed adjacency bitset (n, ceil(n/8)); edge iff Hamming <= thr."""
    n = bits_mat.shape[0]
    P = np.packbits(bits_mat.astype(np.uint8), axis=1)
    A = np.zeros((n, (n + 7) // 8), dtype=np.uint8)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = _POP[np.bitwise_xor(P[s:e, None, :], P[None, :, :])].sum(axis=2)
        adj = d <= thr
        adj[np.arange(e - s), np.arange(s, e)] = False      # no self loops
        A[s:e] = np.packbits(adj, axis=1)
    return A


def _popcount_rows(M):
    return _POP[M].sum(axis=-1)


def greedy_clique(A, restarts=16, seed=42):
    """Greedy expansion: repeatedly add the candidate with the most candidate-neighbours."""
    n = A.shape[0]
    deg = _popcount_rows(A)
    rng = np.random.default_rng(seed)
    starts = [int(np.argmax(deg))]
    if restarts > 1:
        order = np.argsort(-deg)[:max(200, restarts)]
        k = min(restarts - 1, len(order))
        starts += [int(v) for v in rng.choice(order, size=k, replace=False)]

    best = []
    for v0 in starts:
        clique = [v0]
        cand = A[v0].copy()
        while True:
            idx = np.nonzero(np.unpackbits(cand)[:n])[0]
            if idx.size == 0:
                break
            scores = _popcount_rows(np.bitwise_and(A[idx], cand))
            v = int(idx[int(np.argmax(scores))])
            clique.append(v)
            cand = np.bitwise_and(cand, A[v])
        if len(clique) > len(best):
            best = clique
    return sorted(best)


def verify_clique(bits_mat, nodes, thr):
    """Every pair inside `nodes` is within `thr` -- the defining property of a clique."""
    if len(nodes) < 2:
        return True
    X = bits_mat[nodes]
    d = (X[:, None, :] != X[None, :, :]).sum(-1)
    np.fill_diagonal(d, 0)
    return int(d.max()) <= thr
