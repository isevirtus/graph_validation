import os
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

# ============================================================
# STAGE S3 (RQ1): Fit the PC regression on the 8 expert-labeled
# calibration scenarios.
# ============================================================
# Input:  ../00_data/expert_scenarios_8.csv  (SLF, OSF, FS_real)
# Output: outputs/regression_coefficients.csv
#         (per-scenario predictions + residuals)
#         Printed to stdout: Intercept, alpha (SLF), beta (OSF),
#         gamma (SLF*OSF), MAE, Pearson r
#
# Model:
#   PC_base(OSF, SLF) = Intercept + alpha*SLF + beta*OSF + gamma*(SLF*OSF)
#
# f(N) is NOT part of this fit -- the 8 synthetic scenarios have no
# collaboration-history dimension. f(N) is applied afterwards, in
# 02_apply_to_pairs/calculate_pc_pairs.py, as a multiplicative
# saturation factor on top of this base score.
# ============================================================

INPUT_PATH = os.path.join("..", "00_data", "expert_scenarios_8.csv")
OUTPUT_DIR = "outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "regression_coefficients.csv")


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
        # If it parses as-is (dot-decimal or integer), keep it.
        try:
            return float(v)
        except ValueError:
            pass
        # Otherwise, assume Brazilian comma-decimal: '0,42' -> '0.42'
        # (also strips '.' used as a thousands separator, e.g. '1.234,5')
        v2 = v.replace(".", "").replace(",", ".")
        try:
            return float(v2)
        except ValueError:
            return None

    return s.map(clean_one)


def main():
    df = pd.read_csv(INPUT_PATH)

    # Accept alternate column names (e.g. "FS" instead of "FS_real",
    # or a pre-existing "OSF*SLF" interaction column).
    if "FS_real" not in df.columns and "FS" in df.columns:
        df = df.rename(columns={"FS": "FS_real"})
    if "SLF_OSF" not in df.columns and "OSF*SLF" in df.columns:
        df = df.rename(columns={"OSF*SLF": "SLF_OSF"})

    assert {"SLF", "OSF", "FS_real"}.issubset(df.columns), \
        f"CSV must have columns SLF, OSF, FS_real (or FS). Got: {list(df.columns)}"

    for col in ["SLF", "OSF", "FS_real"] + (["SLF_OSF"] if "SLF_OSF" in df.columns else []):
        df[col] = to_numeric_series(df[col])

    print("=== Raw calibration data (after numeric parsing) ===")
    print(df[["SLF", "OSF", "FS_real"] + (["SLF_OSF"] if "SLF_OSF" in df.columns else [])])
    print()

    if df[["SLF", "OSF", "FS_real"]].isna().any().any():
        bad_rows = df[df[["SLF", "OSF", "FS_real"]].isna().any(axis=1)]
        raise ValueError(
            "Some values could not be parsed as numbers after cleaning. "
            f"Check these rows in the CSV:\n{bad_rows}"
        )

    if "SLF_OSF" not in df.columns:
        df["SLF_OSF"] = df["SLF"] * df["OSF"]

    X = sm.add_constant(df[["SLF", "OSF", "SLF_OSF"]])
    y = df["FS_real"]

    model = sm.OLS(y, X).fit()

    df["FS_predicted"] = model.predict(X)
    df["Absolute_Error"] = (df["FS_real"] - df["FS_predicted"]).abs()

    mae = mean_absolute_error(df["FS_real"], df["FS_predicted"])
    pearson_r = df["FS_real"].corr(df["FS_predicted"], method="pearson")

    params = model.params
    print("=== Calibrated PC regression coefficients ===")
    print(f"Intercept        = {params['const']:.6f}")
    print(f"alpha (SLF)      = {params['SLF']:.6f}")
    print(f"beta  (OSF)      = {params['OSF']:.6f}")
    print(f"gamma (SLF*OSF)  = {params['SLF_OSF']:.6f}")
    print()
    print(f"MAE       = {mae:.4f}")
    print(f"Pearson r = {pearson_r:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nOK: saved per-scenario predictions to {OUTPUT_PATH}")

    # also save a clean one-row coefficients file for easy reference
    coeffs_path = os.path.join(OUTPUT_DIR, "regression_coefficients_summary.csv")
    pd.DataFrame([{
        "Intercept": params["const"],
        "alpha_SLF": params["SLF"],
        "beta_OSF": params["OSF"],
        "gamma_SLF_OSF": params["SLF_OSF"],
        "MAE": mae,
        "Pearson_r": pearson_r,
    }]).to_csv(coeffs_path, index=False)
    print(f"OK: saved coefficient summary to {coeffs_path}")


if __name__ == "__main__":
    main()
