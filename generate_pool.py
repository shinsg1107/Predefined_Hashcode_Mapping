"""
Pure biological-constraints primer generator (no Hamming / anchors)
- Filters: GC%, homopolymer max run, Tm window, hairpin, homodimer, heterodimer
- Optional reverse-complement k-mer prefilter to cut obvious cross-priming risks
- Live diagnostics and gentle auto-backoff for Tm / GC if stuck
- Output CSV auto-named by L, N, slack, seed, timestamp
"""

from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

import primer3
from warnings import filterwarnings
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

# Silence deprecation warnings from old primer3 bindings
filterwarnings('ignore', category=UserWarning, module='primer3.bindings')

# -------------------------
# Utilities
# -------------------------
_RC_TBL = str.maketrans("ACGT", "TGCA")

def revcomp(s: str) -> str:
    return s.translate(_RC_TBL)[::-1]

def violates_homopolymer(s: str, max_run: int) -> bool:
    run, prev = 0, None
    for ch in s:
        run = run + 1 if ch == prev else 1
        prev = ch
        if run > max_run:
            return True
    return False

def compute_tm(seq: str, primer_conc_nM: float, na_mM: float) -> float:
    # Biopython NN3 model (Na+ only)
    return float(mt.Tm_NN(Seq(seq), dnac1=primer_conc_nM, Na=na_mM, nn_table=mt.DNA_NN3))

# Reverse-complement prefilters (optional)

def kmer_rc_set_of_seq(s: str, k: int) -> set:
    if k <= 0 or k > len(s):
        return set()
    return {revcomp(s[i:i+k]) for i in range(0, len(s)-k+1)}

def has_rc_kmer_match(cand: str,
                      rc_end3_accepted: set,
                      rc_internal_accepted: set,
                      k_end3: int,
                      k_internal: int) -> bool:
    # 3' k-mer of candidate vs internal of accepted; and candidate internal vs accepted 3'
    if k_end3 > 0:
        end3_rc = revcomp(cand[-k_end3:])
        if end3_rc in rc_internal_accepted:
            return True
    if k_internal > 0:
        for i in range(0, len(cand)-k_internal+1):
            if revcomp(cand[i:i+k_internal]) in rc_end3_accepted:
                return True
    return False

# Primer3-based structure checks (underscore API)

def hairpin_ok(seq: str,
               na_mM: float, mg_mM: float, dntp_mM: float,
               primer_conc_nM: float, temp_c: float,
               dg_thr: float, tm_margin: float) -> bool:
    hp = primer3.calc_hairpin(seq, mv_conc=na_mM, dv_conc=mg_mM, dntp_conc=dntp_mM,
                              dna_conc=primer_conc_nM, temp_c=temp_c)
    dg = getattr(hp, "dg", None)
    tm = getattr(hp, "tm", None)
    if (dg is not None) and (dg < dg_thr):
        return False
    if (tm is not None) and (tm >= temp_c - tm_margin):
        return False
    return True

def homodimer_ok(seq: str,
                 na_mM: float, mg_mM: float, dntp_mM: float,
                 primer_conc_nM: float, temp_c: float,
                 dg_thr: float, tm_margin: float, end3_thr: float) -> bool:
    sd = primer3.calc_homodimer(seq, mv_conc=na_mM, dv_conc=mg_mM, dntp_conc=dntp_mM,
                                dna_conc=primer_conc_nM, temp_c=temp_c)
    dg = getattr(sd, "dg", None)
    tm = getattr(sd, "tm", None)
    dg_end3 = getattr(sd, "dg_end3", None)
    if (dg is not None) and (dg < dg_thr):
        return False
    if (tm is not None) and (tm >= temp_c - tm_margin):
        return False
    if (dg_end3 is not None) and (dg_end3 < end3_thr):
        return False
    return True

def heterodimer_ok_sampled(cand: str,
                           accepted: List[str],
                           na_mM: float, mg_mM: float, dntp_mM: float,
                           primer_conc_nM: float, temp_c: float,
                           dg_thr: float, tm_margin: float, end3_thr: float,
                           sample_size: int = 64,
                           rng: Optional[random.Random] = None) -> bool:
    if not accepted:
        return True
    if rng is None:
        rng = random
    sample = rng.sample(accepted, min(sample_size, len(accepted)))
    for other in sample:
        hd = primer3.calc_heterodimer(other, cand,
                                      mv_conc=na_mM, dv_conc=mg_mM, dntp_conc=dntp_mM,
                                      dna_conc=primer_conc_nM, temp_c=temp_c)
        dg = getattr(hd, "dg", None)
        tm = getattr(hd, "tm", None)
        dg_end3 = getattr(hd, "dg_end3", None)
        if (dg is not None and dg < dg_thr) or \
           (tm is not None and tm >= temp_c - tm_margin) or \
           (dg_end3 is not None and dg_end3 < end3_thr):
            return False
    return True

# -------------------------
# Config
# -------------------------

@dataclass
class BioConfig:
    # Target size
    num_candidates: int = 20_000
    seq_length: int = 20

    # Base biochemical constraints
    min_gc: float = 0.45
    max_gc: float = 0.55
    max_run: int = 4
    target_tm: float = 57.0
    tm_tolerance: float = 3.0
    anneal_temp_c: float = 60.0
    primer_conc_nM: float = 500.0
    na_mM: float = 50.0
    mg_mM: float = 0.0
    dntp_mM: float = 0.2

    # Structure thresholds
    hairpin_dg_thr: float = -3.0
    homodimer_dg_thr: float = -6.0
    end3_dg_thr: float = -12.0
    sec_struct_tm_margin: float = 10.0

    # Optional fast prefilters for cross-priming
    cross_end3_prefilter_k: int = 5
    cross_internal_prefilter_k: int = 7
    allow_end3_collision: bool = False

    # Cross-dimer audit among generated primers
    enable_cross_dimer_audit: bool = True
    cross_dimer_sample_size: int = 64

    # Slack/backoff & diagnostics
    slack: float = 0.10
    gc_backoff_delta: float = 0.01
    global_backoff_attempts: int = 200_000
    max_backoff_tm_factor: float = 1.5
    show_live_postfix: bool = True
    diag_every: int = 20000
    diag_topk: int = 3

    # IO & RNG
    seed: int = 107
    out_dir: str = "./dataset/code"

    def finalize(self):
        assert 0.0 <= self.slack <= 0.2
        os.makedirs(self.out_dir, exist_ok=True)

# -------------------------
# Core generator (bio-only)
# -------------------------

def generate_primers_bio_only(cfg: BioConfig) -> pd.DataFrame:
    cfg.finalize()

    # Effective constraints with slack
    eff_min_gc = max(0.0, cfg.min_gc * (1 - cfg.slack))
    eff_max_gc = min(1.0, cfg.max_gc * (1 + cfg.slack))
    eff_max_run = cfg.max_run + (1 if cfg.slack >= 0.08 else 0)
    eff_tm_tol  = cfg.tm_tolerance * (1 + cfg.slack)
    eff_struct_margin = max(0.0, cfg.sec_struct_tm_margin * (1 - cfg.slack))
    eff_hp_dg  = cfg.hairpin_dg_thr   * (1 + cfg.slack)
    eff_sd_dg  = cfg.homodimer_dg_thr * (1 + cfg.slack)
    eff_end3   = cfg.end3_dg_thr      * (1 + cfg.slack)

    rng_np = np.random.default_rng(cfg.seed)
    rng_py = random.Random(cfg.seed)
    bases = np.array(list("ACGT"))

    accepted: List[str] = []
    accepted_set: Set[str] = set()

    # Prefilter index sets for cross-priming avoidance
    rc_end3_set = set()
    rc_internal_set = set()

    rej: Dict[str, int] = {k:0 for k in [
        "gc_low","gc_high","homopolymer","end3_prefilter","tm_out",
        "hairpin","homodimer","cross_dimer"
    ]}

    attempts = 0
    pbar = tqdm(total=cfg.num_candidates, desc="Generating primers (bio-only)")

    def _diag_update():
        if not cfg.show_live_postfix:
            return
        items = sorted(rej.items(), key=lambda kv: kv[1], reverse=True)
        top = ",".join(f"{k}:{v}" for k, v in items[:cfg.diag_topk]) if items else "—"
        meta = (f"att:{attempts} acc:{pbar.n} tm±:{eff_tm_tol:.1f} "
                f"GC:[{eff_min_gc:.2f},{eff_max_gc:.2f}]")
        pbar.set_postfix_str(meta + " | " + top)

    while pbar.n < cfg.num_candidates:
        cand = ''.join(rng_np.choice(bases, size=cfg.seq_length))
        attempts += 1

        # 1) GC
        gc_ratio = (cand.count('G') + cand.count('C')) / cfg.seq_length
        if gc_ratio < eff_min_gc:
            rej["gc_low"] += 1; (_diag_update() if (attempts % cfg.diag_every == 0) else None); continue
        if gc_ratio > eff_max_gc:
            rej["gc_high"] += 1; (_diag_update() if (attempts % cfg.diag_every == 0) else None); continue

        # 2) homopolymer
        if violates_homopolymer(cand, eff_max_run):
            rej["homopolymer"] += 1; (_diag_update() if (attempts % cfg.diag_every == 0) else None); continue

        # 3) reverse-complement k-mer prefilter (optional)
        if (cfg.cross_end3_prefilter_k > 0 or cfg.cross_internal_prefilter_k > 0) and (not cfg.allow_end3_collision):
            if has_rc_kmer_match(cand, rc_end3_set, rc_internal_set,
                                 cfg.cross_end3_prefilter_k, cfg.cross_internal_prefilter_k):
                rej["end3_prefilter"] += 1; (_diag_update() if (attempts % cfg.diag_every == 0) else None); continue

        # 4) Tm
        tm_val = compute_tm(cand, cfg.primer_conc_nM, cfg.na_mM)
        if abs(tm_val - cfg.target_tm) > eff_tm_tol:
            rej["tm_out"] += 1; (_diag_update() if (attempts % cfg.diag_every == 0) else None); continue

        # 5) Structure checks
        if not hairpin_ok(cand, cfg.na_mM, cfg.mg_mM, cfg.dntp_mM, cfg.primer_conc_nM, cfg.anneal_temp_c, eff_hp_dg, eff_struct_margin):
            rej["hairpin"] += 1; (_diag_update() if (attempts % cfg.diag_every == 0) else None); continue
        if not homodimer_ok(cand, cfg.na_mM, cfg.mg_mM, cfg.dntp_mM, cfg.primer_conc_nM, cfg.anneal_temp_c, eff_sd_dg, eff_struct_margin, eff_end3):
            rej["homodimer"] += 1; (_diag_update() if (attempts % cfg.diag_every == 0) else None); continue

        # 6) Heterodimer audit against already-accepted primers (sampled)
        if cfg.enable_cross_dimer_audit:
            if not heterodimer_ok_sampled(cand, accepted, cfg.na_mM, cfg.mg_mM, cfg.dntp_mM,
                                          cfg.primer_conc_nM, cfg.anneal_temp_c,
                                          eff_sd_dg, eff_struct_margin, eff_end3,
                                          sample_size=cfg.cross_dimer_sample_size, rng=rng_py):
                rej["cross_dimer"] += 1; (_diag_update() if (attempts % cfg.diag_every == 0) else None); continue

        # Accept
        if cand in accepted_set:
            continue  # ultra-rare, but keep uniqueness
        accepted.append(cand)
        accepted_set.add(cand)
        pbar.update(1)
        _diag_update()

        # Update prefilters with newly accepted primer
        if cfg.cross_end3_prefilter_k > 0:
            rc_end3_set.add(revcomp(cand[-cfg.cross_end3_prefilter_k:]))
        if cfg.cross_internal_prefilter_k > 0:
            rc_internal_set |= kmer_rc_set_of_seq(cand, cfg.cross_internal_prefilter_k)

        # Global backoff if nothing accepted for a very long time
        if attempts % cfg.global_backoff_attempts == 0 and pbar.n == 0:
            old = eff_tm_tol
            eff_tm_tol = min(eff_tm_tol + 0.5, cfg.tm_tolerance*(1+cfg.slack)*cfg.max_backoff_tm_factor)
            # widen GC window a bit
            eff_min_gc = max(0.0, eff_min_gc - cfg.gc_backoff_delta)
            eff_max_gc = min(1.0, eff_max_gc + cfg.gc_backoff_delta)
            print(f"\n[BACKOFF] No accept yet → Tm tol ±{old:.1f}→±{eff_tm_tol:.1f}, GC→[{eff_min_gc:.2f},{eff_max_gc:.2f}]")

    pbar.close()

    # Build DataFrame
    df = pd.DataFrame({"code": accepted})

    # Stats
    print("\n--- Rejection Statistics (bio-only) ---")
    total_rej = sum(rej.values())
    print(f"Attempts: {attempts:,} | Accepted: {len(accepted):,} | Rejected: {total_rej:,}")
    for k, v in rej.items():
        if total_rej > 0:
            print(f"- {k:16s}: {v:10d}  ({v/total_rej:6.2%})")
        else:
            print(f"- {k:16s}: {v:10d}")

    # Output path
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_name = (
        f"primers_bio_only_{cfg.seq_length}nt_{cfg.num_candidates}codes_"
        f"slack{cfg.slack:.2f}_seed{cfg.seed}_{stamp}.csv"
    )
    out_path = os.path.join(cfg.out_dir, out_name)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {len(df)} primers to: {out_path}")

    return df

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    cfg = BioConfig(
        num_candidates=100000,
        seq_length=16,
        min_gc=0.45, max_gc=0.55,
        max_run=4,
        target_tm=57.0, tm_tolerance=3.0,
        anneal_temp_c=60.0,
        primer_conc_nM=500.0, na_mM=50.0, mg_mM=0.0, dntp_mM=0.2,
        hairpin_dg_thr=-3.0, homodimer_dg_thr=-6.0, end3_dg_thr=-12.0,
        sec_struct_tm_margin=10.0,
        cross_end3_prefilter_k=5, cross_internal_prefilter_k=7,
        allow_end3_collision=False,
        enable_cross_dimer_audit=True, cross_dimer_sample_size=64,
        slack=0.10,
        gc_backoff_delta=0.01, global_backoff_attempts=200_000, max_backoff_tm_factor=1.5,
        show_live_postfix=True, diag_every=20000, diag_topk=3,
        seed=107,
        out_dir="./dataset/codeset",
    )

    # cfg = BioConfig(
    # num_candidates=100000,
    # seq_length=32,
    # min_gc=0.42, max_gc=0.58,
    # max_run=4,                 # 막히면 5로
    # target_tm=68.0,            # 66~70 근처가 보통 무난
    # tm_tolerance=6.0,          # ±6
    # anneal_temp_c=62.0,        
    # primer_conc_nM=500.0,
    # na_mM=50.0,
    # mg_mM=0.0,
    # dntp_mM=0.2,
    # sec_struct_tm_margin=10.0,
    # cross_end3_prefilter_k=6,
    # cross_internal_prefilter_k=8,
    # allow_end3_collision=False,
    # enable_cross_dimer_audit=True,  # 막히면 False로 끄거나 샘플 수 줄여 속도/수락률 ↑
    # cross_dimer_sample_size=32,
    # slack=0.15,              
    # gc_backoff_delta=0.02,    
    # global_backoff_attempts=100_000,
    # max_backoff_tm_factor=1.5,
    # show_live_postfix=True, diag_every=20000, diag_topk=3,
    # seed=107,
    # out_dir="./dataset/codeset",
    # )


    _ = generate_primers_bio_only(cfg)
