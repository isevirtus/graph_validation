# 🧪 Developer Compatibility Graph Validation

This repository provides a validation framework for estimating developer compatibility based on historical collaboration data. The goal is to calibrate a compatibility model aligned with expert expectations and to generate a collaboration graph usable in team formation algorithms.

---

## 📦 Repository Overview

This repository contains:

- `src/`: Python scripts for compatibility score calculation, regression fitting, graph generation, and transformation of data.
- `data/`: Input/output files used in the validation process.
- `expert_calibration_data`: Master spreadsheet containing expert inputs, heatmaps, probabilistic distributions, PC calculations, and the FS regression summary.
- `PC_levels_Definitions.pdf`: Expert-validated definitions and qualitative criteria for each PC label (e.g., PC_VL, PC_H), with examples and rationale.

---

## 📂 Repository Structuring

```
graph_validation/
├── data/
│   ├── dataset.json
│   ├── fs_expert_input.csv.csv
│   ├── fs_regression_summary.txt
│   ├── network_edges_details_ordenado1.csv
│   ├── network_plot.png
│   ├── parsed_graph.json
│   ├── regression_plot.png
│   ├── regression_results_fs.csv
│   └── report_pc.txt
│
├── src/
│   ├── calculate_pc.py
│   ├── fit_fs_regression.py
│   ├── plot_graph.py
│   └── transformation_graph_data.py
│
├── expert_calibration_data.xlsx
└──  PC_Levels_Definitions.pdf
```

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/isevirtus/graph_validation.git
   cd graph_validation
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn networkx
   ```

---

## 🚀 Kickstart

Run the validator for each component in the pipeline:

### 1. Fit the regression model:
```bash
python src/fit_fs_regression.py
```

### 2. Calculate PC scores from FS predictions:
```bash
python src/calculate_pc.py
```

### 3. Generate and export the graph:
```bash
python src/plot_graph.py
```

All outputs will be saved in the `data/` folder.

---

## 📊 About the Spreadsheet (`excel_decalibration.xlsx`)

This file contains all expert-driven calibration data used in the validation:

- ✔️ Input scenarios for FS regression
- 🔥 Heatmaps of FS and PC
- 📈 Intermediate distributions and formulas
- 🧮 PC value calculations and category conversion (e.g., PC_VH, PC_H, etc.)
- 🧾 Regression coefficients and residuals

---

## 📘 About the Definitions Document (`PC_Livres_Definitions.pdf`)

This PDF provides formal definitions and qualitative descriptions for each PC category:

- Meaning of each label (Very Low, Low, Medium, etc.)
- Behavioral characteristics associated with each level
- Real-world examples drawn from validated collaboration scenarios
- Validation methodology used with domain experts

This document supports the semantic interpretability of the compatibility labels used in the graph.

---

## 🧠 Objective

This repository implements a validation framework to align a data-driven compatibility graph with expert-derived expectations. The resulting graph expresses the collaborative potential between developer pairs and supports intelligent team formation.

---

## 📄 License

MIT License.
