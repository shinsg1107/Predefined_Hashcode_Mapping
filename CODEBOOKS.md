# Codebooks

This document describes the code representation, refinement settings, and
generation procedure used for the released PHM codebooks.

## 1. Code Representation

`tools.one_hot_encode` maps each nucleotide to a bipolar pair:

| Base | Bipolar vector | Binary sign pattern |
|---|---|---|
| A | `(-1, -1)` | `00` |
| T | `(-1, +1)` | `01` |
| C | `(+1, -1)` | `10` |
| G | `(+1, +1)` | `11` |

A code of length \(L\) is represented as a \(2L\)-dimensional bipolar
vector. The corresponding binary sign pattern is used for the Hamming
distances in pseudo-labeling, refinement, and retrieval.

## 2. Released Set 1 Codebooks

The following codebooks are provided under `codebooks/`. A codebook is
shared across backbones for the same dataset and code length.

| Dataset | Length | Codes | `tau_in` | `tau_out` | `m` | File |
|---|---:|---:|---:|---:|---:|---|
| CIFAR-10 | 16 nt | 3,848 | 12 | 3 | 0 | `set1_cifar10_16nt.csv` |
| CIFAR-10 | 32 nt | 938 | 32 | 16 | 0 | `set1_cifar10_32nt.csv` |
| ImageNet-100 | 16 nt | 9,148 | 12 | 3 | 0 | `set1_imagenet_16nt.csv` |
| ImageNet-100 | 32 nt | 7,676 | 34 | 16 | 0 | `set1_imagenet_32nt.csv` |
| NUS-WIDE | 16 nt | 5,657 | 12 | 3 | 0 | `set1_nuswide_16nt.csv` |
| NUS-WIDE | 32 nt | 3,315 | 34 | 16 | 0 | `set1_nuswide_32nt.csv` |

With `m = 0`, a code is removed if it has any cross-label neighbor within
`tau_out`.

## 3. Generation Procedure

Codebook generation consists of three stages:

1. **OT pseudo-labeling.** Candidate codes are assigned to the fixed anchor
   set using `tools.code_ot_mapping`.
2. **Within-label refinement.** For each pseudo-label, `clique.py` applies a
   deterministic greedy approximation to the maximum-clique objective under
   `tau_in`.
3. **External-margin filtering.** Codes with more than `m` cross-label
   neighbors within `tau_out` are removed.

The following command regenerates the CIFAR-10 16-nt codebook from the
included 16-nt candidate pool:

```bash
python gen_codebook_tl.py \
    --pool dataset/codeset/pool_16nt_100k.csv \
    --anchors dataset/codeset/anchor_codes_10.csv \
    --num-label 10 \
    --tau-in 12 \
    --tau-out 3 \
    --max-bad 0 \
    --device cuda:0 \
    --out codebooks/my_cifar10_16nt.csv
```

For ImageNet-100 and NUS-WIDE, use the corresponding anchor file and class
count:

| Dataset | Anchor file | Classes |
|---|---|---:|
| CIFAR-10 | `dataset/codeset/anchor_codes_10.csv` | 10 |
| ImageNet-100 | `dataset/codeset/anchor_codes_100.csv` | 100 |
| NUS-WIDE | `dataset/codeset/16anchor_codes_21.csv` | 21 |

The script reports per-label clique sizes, the number of codes removed by
the external-margin stage, and the final codebook cardinality.

## 4. Set 2

`dataset/codeset/set2_16nt_20k.csv` and
`dataset/codeset/set2_32nt_20k.csv` contain 20,000 held-out codes each.
They are not used during PHM training. At evaluation time, pseudo-labels are
assigned by OT and the trained hash encoder is evaluated without further
optimization.

## 5. Candidate Pools

The biologically admissible 16-nt candidate pool used by the generator is
included under `dataset/codeset/`. It was produced with `generate_pool.py`
using the biological constraints described in the manuscript.

The released 32-nt codebooks are included in `codebooks/`. Regenerating
them from scratch requires the corresponding 32-nt candidate pool.
