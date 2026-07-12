import sys, os
import numpy as np
import pandas as pd

ROOT = r"D:\DOCUMENTS\works\STFP"
sys.path.insert(0, ROOT)

from Algorithms.BN.team_fit_bn import criar_rede_fitness

DATA = os.path.join(os.path.dirname(__file__), "00_data", "rq3_scenarios.csv")
OUT  = os.path.join(os.path.dirname(__file__), "rq3_results.csv")

def brier(p_pred, p_true):
    return float(np.mean((np.array(p_pred) - np.array(p_true)) ** 2))

def main():
    bn = criar_rede_fitness(
        func_at="WMEAN",     sigma_at=0.5,  pesos_dom_eco_ling=[3, 1, 5],
        func_ae="MIXMINMAX", sigma_ae=0.5,  pesos_at_ac=[5, 1],
    )

    df = pd.read_csv(DATA)
    expert_cols = ["ae_expert_vl","ae_expert_l","ae_expert_m","ae_expert_h","ae_expert_vh"]

    rows = []
    for _, row in df.iterrows():
        bn.evidence_distributions.clear()
        bn.evidence_hard.clear()
        bn.setEvidence("AT", row["AT"])
        bn.setEvidence("AC", row["AC"])
        pred = bn.calculateTPN("AE")
        true = row[expert_cols].astype(float).to_numpy()
        b = brier(pred, true)
        rows.append({
            "scenario_id": row["scenario_id"],
            "split": row["split"],
            "AT": row["AT"], "AC": row["AC"],
            "pred_vl": round(pred[0],4), "pred_l": round(pred[1],4),
            "pred_m":  round(pred[2],4), "pred_h": round(pred[3],4),
            "pred_vh": round(pred[4],4),
            "brier": round(b, 6),
        })
        print(f"{row['scenario_id']:12} AT={row['AT']:3} AC={row['AC']:3} "
              f"pred={[round(x,4) for x in pred]} brier={b:.4f} [{row['split']}]")

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT, index=False)

    test  = rdf[rdf["split"]=="test"]
    train = rdf[rdf["split"]=="train"]
    print(f"\n=== RESULTADO FINAL ===")
    print(f"TEST  briers: {test['brier'].tolist()}")
    print(f"TEST  media:  {test['brier'].mean():.6f}")
    print(f"TRAIN media:  {train['brier'].mean():.6f}")
    print(f"Resultados salvos em: {OUT}")

if __name__ == "__main__":
    main()