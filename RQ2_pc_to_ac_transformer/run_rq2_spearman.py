"""
RQ2 -- Intra-team collaboration representation (ranking-based evaluation)

Compares three representations of intra-team collaboration against the expert's
expected AC value, using Spearman rank correlation:
  - C1-Weighted: PC-tier distribution aggregated with weights (1,1,4,4,5)
  - C2-Mean: arithmetic mean of pairwise PC
  - Majority-tier baseline: centroid of the most frequent tier

Input CSV must contain, per scenario:
  - tier proportions: pc_p_vl, pc_p_l, pc_p_m, pc_p_h, pc_p_vh
  - expert AC distribution: ac_expert_vl, ac_expert_l, ac_expert_m, ac_expert_h, ac_expert_vh
  - (optional) pc_mean : arithmetic mean of pairwise PC for that scenario

Usage:
  python run_rq2_spearman.py --csv 00_data\\rq2_inputs.csv --out outputs\\rq2_spearman_results.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CENTROIDS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
TIER_COLS = ["pc_dist_PC_VL", "pc_dist_PC_L", "pc_dist_PC_M", "pc_dist_PC_H", "pc_dist_PC_VH"]
EXPERT_COLS = ["ac_expert_vl", "ac_expert_l", "ac_expert_m", "ac_expert_h", "ac_expert_vh"]
PESOS_PC = (1, 1, 4, 4, 5)


def pc_score_from_props(props):
    """C1-Weighted: tier proportions -> single score in [0.1, 0.9]."""
    s = props.sum()
    if s <= 0:
        return 0.5
    p = props / s
    raw = float(np.dot(PESOS_PC, p))          # [1, 5]
    return 0.1 + 0.8 * ((raw - 1.0) / 4.0)    # [0.1, 0.9]


def expected_value(dist):
    """Centroid-weighted mean of a 5-state distribution."""
    d = np.asarray(dist, dtype=float)
    s = d.sum()
    if s <= 0:
        return 0.5
    return float(np.dot(d / s, CENTROIDS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    for c in TIER_COLS + EXPERT_COLS:
        if c not in df.columns:
            raise KeyError(f"Coluna obrigatoria ausente: {c}")

    rows = []
    for _, r in df.iterrows():
        props = r[TIER_COLS].to_numpy(dtype=float)
        expert = r[EXPERT_COLS].to_numpy(dtype=float)

        ev_expert = expected_value(expert)
        c1 = pc_score_from_props(props)

        # C2-Mean: usa coluna pc_mean se existir, senao calcula do tier profile
        if "pc_mean" in df.columns and not pd.isna(r.get("pc_mean")):
            c2 = float(r["pc_mean"])
        else:
            s = props.sum()
            c2 = float(np.dot(props / s, CENTROIDS)) if s > 0 else 0.5

        # Majority-tier baseline
        moda = float(CENTROIDS[int(np.argmax(props))])

        rows.append({
            "scenario_id": r.get("scenario_id", ""),
            "ev_expert": round(ev_expert, 4),
            "c1_weighted": round(c1, 4),
            "c2_mean": round(c2, 4),
            "majority_tier": round(moda, 4),
        })

    res = pd.DataFrame(rows)

    rho_c1, p_c1 = spearmanr(res["c1_weighted"], res["ev_expert"])
    rho_c2, p_c2 = spearmanr(res["c2_mean"], res["ev_expert"])
    rho_mj, p_mj = spearmanr(res["majority_tier"], res["ev_expert"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res.to_csv(args.out, index=False)

    print(res.to_string(index=False))
    print("\n=== RQ2 -- Spearman rank correlation vs expert expected AC ===")
    print(f"  C1-Weighted (tier distribution): rho = {rho_c1:.4f}, p = {p_c1:.6f}")
    print(f"  C2-Mean (arithmetic mean):       rho = {rho_c2:.4f}, p = {p_c2:.6f}")
    print(f"  Majority-tier baseline:          rho = {rho_mj:.4f}, p = {p_mj:.6f}")
    print(f"\nResultados salvos em: {args.out}")


if __name__ == "__main__":
    main()
