import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

# === 1. Expert-provided data: SLF, OSF, FS_real (calculated from VL–VH label weights) ===
data = {
    "SLF": [0.9, 0.1, 0.1, 0.9, 0.5, 0.1, 0.5, 0.9],
    "OSF": [0.9, 0.9, 0.1, 0.9, 0.1, 0.5, 0.9, 0.5],
    "FS_real": [0.42, 0.348, 0.224137931, 0.6104477612, 0.2657142857, 0.5473684211, 0.63, 0.644]
}

df = pd.DataFrame(data)

# === 2. Create the interaction variable ===
df["SLF_OSF"] = df["SLF"] * df["OSF"]

# === 3. Prepare X and y for regression ===
X = df[["SLF", "OSF", "SLF_OSF"]]
X = sm.add_constant(X)  # add constant term (intercept)
y = df["FS_real"]

# === 4. Fit the regression model ===
model = sm.OLS(y, X).fit()

# === 5. Predict FS using the calibrated formula ===
df["FS_predicted"] = model.predict(X)

# === 6. Calculate Mean Absolute Error (MAE) ===
mae = mean_absolute_error(df["FS_real"], df["FS_predicted"])
print(f"MAE = {mae:.4f}")

# === 7. Display regression coefficients ===
print("\nRegression coefficients:")
print(model.params)

# === 8. Compute absolute error for each row ===
df["Absolute_Error"] = abs(df["FS_real"] - df["FS_predicted"])

# === 9. Export results to CSV ===
output_file = "regression_results_fs.csv"
df.to_csv(output_file, index=False)
print(f"\nFile saved as: {output_file}")
