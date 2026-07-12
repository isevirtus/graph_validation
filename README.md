# Supplementary Material

## An Expert-Guided Method for Developer Compatibility Graphs and Bayesian Team-Fit Evaluation

This package contains the data, scripts, and calibrated model configurations required to reproduce the experimental results reported in the paper (RQ1, RQ2, and RQ3). No sensitive or personally identifiable information is included; all developer records are anonymized or synthetically instantiated within expert-validated ranges.

---

## 1. Package Structure

```
supplementary_material/
|-- README.md                          This file
|-- requirements.txt                   Python dependencies
|-- Algorithms/
|   |-- BN/
|       |-- team_fit_bn.py             Bayesian Network builder (calibrated configuration)
|       |-- bnetwork.py                BN wrapper over pgmpy (evidence handling, inference)
|       |-- utils.py                   RNM/CPT generation utilities (TNormal mixtures)
|       |-- repositorio.json           Pre-generated TNormal sample repository (100k samples per state)
|-- RQ1_pc_regression/
|   |-- 00_data/
|   |   |-- expert_scenarios_8.csv     8 expert-labeled calibration scenarios (OSF, SLF, PC)
|   |   |-- developers_base.json       Semi-synthetic developer base (192 developers)
|   |-- 01_calibration/
|   |   |-- fit_pc_regression.py       Fits the PC regression on the 8 expert scenarios
|   |   |-- loocv_pc_regression.py     Leave-one-out cross-validation of the regression
|   |   |-- outputs/                   Fitted coefficients and LOOCV results
|   |-- 02_apply_to_pairs/
|   |   |-- calculate_pc_pairs.py      Applies the calibrated model to all developer pairs
|   |   |-- outputs/
|   |       |-- pc_pairs_report.txt    Pairwise PC computation report
|   |-- 03_graph/
|       |-- build_graph.py             Builds the weighted compatibility graph from retained pairs
|       |-- plot_graph.py              Renders the graph visualizations reported in the paper
|       |-- outputs/
|           |-- compatibility_graph.json
|           |-- network_edges.csv
|           |-- network_plot_full.png
|           |-- network_plot_top_subgraph.png
|-- RQ2_pc_to_ac_transformer/
|   |-- 00_data/
|   |   |-- rq2_inputs.csv             15 expert-labeled collaboration scenarios
|   |-- run_rq2_spearman.py            RQ2 evaluation script (Spearman rank correlation)
|   |-- outputs/
|       |-- rq2_spearman_results.csv   Per-scenario scores
|-- RQ3_bn/
    |-- 00_data/
    |   |-- rq3_scenarios.csv          Expert-provided AE validation scenarios (2 held-out + 6 calibration)
    |-- run_rq3.py                     RQ3 evaluation script (Brier Score)
    |-- rq3_results.csv                Generated results
```

---

## 2. Environment Setup

### Requirements

- Python 3.10 or later (tested with Python 3.12 and 3.13)
- Operating system: Windows, Linux, or macOS

### Installation

From the root of this package:

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Dependencies installed: numpy, pandas, scipy, pgmpy.

---

## 3. Reproducing RQ1 (Pair Compatibility Modeling)

RQ1 tests whether the regression-calibrated PC model approximates expert expectations of pairwise collaboration quality from OSF, SLF, and a collaboration-history saturation term.

### Step 1: Fit the regression

```bash
cd RQ1_pc_regression/01_calibration
python fit_pc_regression.py
```

This fits the PC model on the 8 expert-labeled scenarios in `00_data/expert_scenarios_8.csv`. Expected fitted coefficients and in-sample fit:

- Intercept = 0.2302, alpha (SLF) = 0.1792, beta (OSF) = 0.2307, gamma (saturation) = 0.1040
- MAE = 0.0812, Pearson r = 0.80 against the expert-provided expected PC values

### Step 2: Leave-one-out cross-validation

```bash
python loocv_pc_regression.py
```

Reports the out-of-sample estimate on the 8 calibration scenarios (LOOCV MAE approximately 0.176), as discussed in the paper's Threats to Validity.

### Step 3: Apply the model to all developer pairs

```bash
cd ../02_apply_to_pairs
python calculate_pc_pairs.py
```

Applies the calibrated model to the semi-synthetic developer base (`00_data/developers_base.json`, 192 developers). After the PC > 0.3 filter, 773 valid developer pairs are retained. A summary is written to `outputs/pc_pairs_report.txt`.

### Step 4: Generate the compatibility graph

```bash
cd ../03_graph
python build_graph.py
python plot_graph.py
```

`build_graph.py` builds the weighted developer compatibility graph from the retained pairs and exports it as `outputs/compatibility_graph.json` (graph structure) and `outputs/network_edges.csv` (edge list with PC weights). `plot_graph.py` renders the visualizations reported in the paper: `outputs/network_plot_full.png` (full graph, 192 nodes and 773 edges) and `outputs/network_plot_top_subgraph.png` (top-connectivity subgraph highlighting cohesive collaboration clusters and bridge developers).

---

## 4. Reproducing RQ2 (Intra-Team Collaboration Representation)

RQ2 tests whether representing intra-team collaboration as a distribution of pairwise compatibility tiers ranks collaboration quality in closer agreement with expert judgment than collapsing it into a single mean value.

### Input data

`RQ2_pc_to_ac_transformer/00_data/rq2_inputs.csv` contains 15 expert-labeled scenarios. Each row provides:

- `pc_dist_PC_VL ... pc_dist_PC_VH`: the proportion of developer pairs in each compatibility tier;
- `ac_expert_vl ... ac_expert_vh`: the expert-elicited AC probability distribution over the five ordinal states;
- `pc_mean`: the arithmetic mean of pairwise PC values (baseline input).

### Execution

```bash
cd RQ2_pc_to_ac_transformer
python run_rq2_spearman.py --csv 00_data/rq2_inputs.csv --out outputs/rq2_spearman_results.csv
```

### Expected output

```
C1-Weighted (tier distribution): rho = 0.8818, p = 0.000014
C2-Mean (arithmetic mean):       rho = 0.8117, p = 0.000238
Majority-tier baseline:          rho = 0.8144, p = 0.000219
```

These values correspond to the RQ2 Spearman table in the paper (rho = 0.882 for the tier-distribution transformer versus 0.812 for the mean-PC baseline). Per-scenario scores are written to `outputs/rq2_spearman_results.csv`.

### Method summary

For each scenario, the expert's AC distribution is reduced to its expected value using the tier centroids (0.1, 0.3, 0.5, 0.7, 0.9). Each representation produces a scalar score on the same scale: the C1-Weighted transformer aggregates tier proportions with the calibrated weights (1,1,4,4,5) and rescales to [0.1, 0.9]; the C2-Mean baseline uses the arithmetic mean of pairwise PC values; the majority-tier baseline uses the centroid of the most frequent tier. The Spearman rank correlation between each representation's scores and the expert's expected values, across the 15 scenarios, measures ordinal agreement.

---

## 5. Reproducing RQ3 (BN Predictive Accuracy for Team Fit)

RQ3 tests whether the expert-calibrated Bayesian Network generalizes to previously unseen AT/AC configurations when predicting Team Fit (AE).

### Input data

`RQ3_bn/00_data/rq3_scenarios.csv` contains the validation scenarios. Each row provides:

- `AT`, `AC`: ordinal evidence levels (VL, L, M, H, VH) injected into the BN;
- `ae_expert_vl ... ae_expert_vh`: the expert-provided target AE distribution;
- `split`: `test` for the two held-out scenarios reported in the paper, `train` for the six calibration scenarios (reference only).

### Calibrated configuration

The BN is built by `Algorithms/BN/team_fit_bn.py` with the configuration reported in the paper:

- AT: WMEAN with weights (3, 1, 5) over Domain, Ecosystem, Language; sigma = 0.05
- AE: MIXMINMAX with weights (5, 1) over the (min, max) of AT and AC; sigma = 0.05
- CPTs generated via the Ranked Nodes Method (RNM) using truncated-normal mixtures sampled from `repositorio.json`

### Execution

```bash
cd RQ3_bn
python run_rq3.py
```

The first run builds the CPTs (approximately 5 to 10 seconds). Results are written to `rq3_results.csv`.

### Expected output

```
S_test_1   AT=VH  AC=M   brier=0.078790 [test]
S_test_2   AT=M   AC=VH  brier=0.078186 [test]
```

These are the two held-out Brier Scores reported in the paper (0.0788 and 0.0782). The six `train` rows reproduce the calibration scenarios and are provided for reference.

Note: identical predicted distributions for the two test scenarios are expected behavior. MIXMINMAX operates on the minimum and maximum of its inputs, and both scenarios share the same input pair {M, VH}; the Brier Scores differ only because the expert target distributions differ.

---

## 6. Reproducibility Notes

- All computations are deterministic given the provided `repositorio.json` sample repository; no random seeds are required at evaluation time.
- All scripts use relative paths and can be executed from their own directories as shown above.
- Expected total runtime: under two minutes for all three experiments on a standard laptop.
- All expert-elicited data included here was collected in separate elicitation sessions, as described in Section 4 of the paper.

## 7. Contact

For questions about this material, please contact the corresponding author.
