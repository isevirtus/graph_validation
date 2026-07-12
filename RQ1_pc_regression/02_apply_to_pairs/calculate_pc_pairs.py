import json
import itertools
import os
import numpy as np

# ============================================================
# STAGE S4 (RQ1): Apply the calibrated PC regression to real
# developer pairs from the synthetic base.
# ============================================================
# Input:  00_data/developers_base.json  (base_sintetica_devs.json)
# Output: 02_apply_to_pairs/outputs/pc_pairs_report.txt
#         (same text format as the old report_pc.txt, so
#          03_graph/build_graph.py keeps working unchanged)
#
# Formula used here (matches Table 3 / regression_coefficients.csv,
# i.e. the model actually fit in 01_calibration):
#
#   PC_base(OSF, SLF) = intercept + alpha*SLF + beta*OSF + gamma*(SLF*OSF)
#   PC_final           = PC_base * f(N)
#
# f(N) is applied as a MULTIPLICATIVE saturation penalty on top of the
# fitted OSF/SLF model, kept separate from the regression itself,
# since the 8 calibration scenarios have no collaboration-history (N)
# dimension and therefore could not have been used to fit a joint
# OSF*SLF*f(N) interaction term.
#


INPUT_PATH = os.path.join("..", "00_data", "developers_base.json")
OUTPUT_DIR = "outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pc_pairs_report.txt")

# --- Calibrated coefficients (Table 3 / regression_coefficients.csv) ---
INTERCEPT = 0.230232
ALPHA_SLF = 0.179206
BETA_OSF = 0.230728
GAMMA_INT = 0.104040  # SLF * OSF interaction

# --- Zero-history business rule ("regra do Sergio", Low tier) ---
INCLUDE_ZERO_HISTORY = True
ZERO_HISTORY_PC = 0.30
CLAMP_01 = True


def f_N(N: int) -> float:
    """Saturation function (Eq. 1 in the paper)."""
    if N <= 0:
        return 0.0
    if N <= 4:
        return N / 4.0
    return max(0.0, 1.0 - 0.1 * (N - 4))


def pc_base(osf_avg: float, slf_avg: float) -> float:
    return (
        INTERCEPT
        + ALPHA_SLF * slf_avg
        + BETA_OSF * osf_avg
        + GAMMA_INT * (slf_avg * osf_avg)
    )


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    devs = data if isinstance(data, list) else data.get("developers", [])

    dev_projects = {}
    for dev in devs:
        dev_id = dev.get("id")
        projs = dev.get("projects") or []
        if not dev_id or not projs:
            continue
        dev_projects[dev_id] = {
            p["id"]: {"osf": p["osf"], "slf": p["slf"]}
            for p in projs
            if p.get("id") is not None
        }

    report_lines = []
    seen = set()
    n_zero_history = 0
    n_with_history = 0

    for d1, d2 in itertools.combinations(dev_projects.keys(), 2):
        pair = tuple(sorted((d1, d2)))
        if pair in seen:
            continue
        seen.add(pair)

        p1 = dev_projects[d1]
        p2 = dev_projects[d2]
        common = set(p1.keys()) & set(p2.keys())
        N = len(common)

        if N == 0:
            if INCLUDE_ZERO_HISTORY:
                n_zero_history += 1
                report_lines.append(
                    f"\n\U0001f539 Developer Pair: ID={d1} e ID={d2}\n"
                    f"\U0001f4c2 Common Projects: None\n"
                    f"\U0001f4ca Avg OSF: 0.000 | Avg SLF: 0.000\n"
                    f"\U0001f522 N = 0 -> f(N) = 0.000\n"
                    f"\U0001f3af Final PC = {ZERO_HISTORY_PC:.3f}\n"
                )
            continue

        osfs = [(p1[pid]["osf"] + p2[pid]["osf"]) / 2 for pid in common]
        slfs = [(p1[pid]["slf"] + p2[pid]["slf"]) / 2 for pid in common]
        osf_avg = float(np.mean(osfs))
        slf_avg = float(np.mean(slfs))
        fn = f_N(N)

        pc = pc_base(osf_avg, slf_avg) * fn
        if CLAMP_01:
            pc = max(0.0, min(1.0, pc))

        n_with_history += 1
        report_lines.append(
            f"\n\U0001f539 Developer Pair: ID={d1} e ID={d2}\n"
            f"\U0001f4c2 Common Projects: {sorted(common)}\n"
            f"\U0001f4ca Avg OSF: {osf_avg:.3f} | Avg SLF: {slf_avg:.3f}\n"
            f"\U0001f522 N = {N} -> f(N) = {fn:.3f}\n"
            f"\U0001f3af Final PC = {pc:.3f}\n"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"OK: PC report saved to {OUTPUT_PATH}")
    print(f"   Total developers with projects: {len(dev_projects)}")
    print(f"   Total pairs processed: {len(seen)}")
    print(f"   Pairs with N=0 (default PC={ZERO_HISTORY_PC}): {n_zero_history}")
    print(f"   Pairs with N>=1 (model-based PC): {n_with_history}")
    print(f"   Pairs with PC > 0.3 (graph-eligible, S5 rule): "
          f"count manually or extend this script once you confirm the S4/S5 filter.")


if __name__ == "__main__":
    main()
