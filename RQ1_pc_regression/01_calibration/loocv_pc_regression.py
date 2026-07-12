import os
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# ============================================================
# STAGE S3 (RQ1) - Leave-One-Out Cross-Validation
# ============================================================
# Input:  ../00_data/expert_scenarios_8.csv  (SLF, OSF, FS_real)
# Output: outputs/loocv_results.csv
# ============================================================

INPUT_PATH = os.path.join("..", "00_data", "expert_scenarios_8.csv")
OUTPUT_DIR = "outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "loocv_results.csv")


def to_numeric_series(s: pd.Series) -> pd.Series:
    """Robust numeric parsing: handles Brazilian comma-decimals and stray
    whitespace (e.g. '0,42' -> 0.42), without touching values that are
    already valid dot-decimal numbers."""
    if pd.api.types.is_numeric_dtype(s):
        return s

    def clean_one(v):
        v = str(v).strip()
        if v == "" or v.lower() == "nan":
            return None
        try:
            return float(v)
        except ValueError:
            pass
        v2 = v.replace(".", "").replace(",", ".")
        try:
            return float(v2)
        except ValueError:
            return None

    return s.map(clean_one)


def main():
    df = pd.read_csv(INPUT_PATH)

    if "FS_real" not in df.columns and "FS" in df.columns:
        df = df.rename(columns={"FS": "FS_real"})
    if "SLF_OSF" not in df.columns and "OSF*SLF" in df.columns:
        df = df.rename(columns={"OSF*SLF": "SLF_OSF"})

    assert {"SLF", "OSF", "FS_real"}.issubset(df.columns), \
        f"CSV must have columns SLF, OSF, FS_real (or FS). Got: {list(df.columns)}"

    for col in ["SLF", "OSF", "FS_real"] + (["SLF_OSF"] if "SLF_OSF" in df.columns else []):
        df[col] = to_numeric_series(df[col])

    if df[["SLF", "OSF", "FS_real"]].isna().any().any():
        bad_rows = df[df[["SLF", "OSF", "FS_real"]].isna().any(axis=1)]
        raise ValueError(
            "Some values could not be parsed as numbers after cleaning. "
            f"Check these rows in the CSV:\n{bad_rows}"
        )

    if "SLF_OSF" not in df.columns:
        df["SLF_OSF"] = df["SLF"] * df["OSF"]

    loocv_rows = []
    for i in range(len(df)):
        train_idx = [j for j in range(len(df)) if j != i]
        train_df = df.iloc[train_idx]
        test_df = df.iloc[[i]]

        X_train = sm.add_constant(train_df[["SLF", "OSF", "SLF_OSF"]])
        y_train = train_df["FS_real"]
        model = sm.OLS(y_train, X_train).fit()

        X_test = sm.add_constant(test_df[["SLF", "OSF", "SLF_OSF"]], has_constant="add")
        y_pred = model.predict(X_test).values[0]

        loocv_rows.append({
            "scenario_id": df.iloc[i].get("scenario_id", f"S{i+1}"),
            "SLF": df.iloc[i]["SLF"],
            "OSF": df.iloc[i]["OSF"],
            "FS_real": df.iloc[i]["FS_real"],
            "FS_predicted_LOOCV": y_pred,
        })

    loocv_df = pd.DataFrame(loocv_rows)
    loocv_df["Absolute_Error"] = (loocv_df["FS_real"] - loocv_df["FS_predicted_LOOCV"]).abs()

    # Full-data model (for comparison against LOOCV)
    X_full = sm.add_constant(df[["SLF", "OSF", "SLF_OSF"]])
    y_full = df["FS_real"]
    full_model = sm.OLS(y_full, X_full).fit()
    df["FS_predicted_final"] = full_model.predict(X_full)

    mae_fit = mean_absolute_error(df["FS_real"], df["FS_predicted_final"])
    r_fit, p_fit = pearsonr(df["FS_real"], df["FS_predicted_final"])

    mae_loocv = mean_absolute_error(loocv_df["FS_real"], loocv_df["FS_predicted_LOOCV"])
    r_loocv, p_loocv = pearsonr(loocv_df["FS_real"], loocv_df["FS_predicted_LOOCV"])

    print("=== Fitting (full 8-point model) ===")
    print(f"MAE = {mae_fit:.4f} | Pearson r = {r_fit:.4f} (p={p_fit:.4f})")
    print()
    print("=== LOOCV (generalization) ===")
    print(f"MAE = {mae_loocv:.4f} | Pearson r = {r_loocv:.4f} (p={p_loocv:.4f})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    loocv_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nOK: saved LOOCV results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
