# Reproduction Journal — PPORank

**Paper**: PPORank (see `paper.pdf`)  
**Original repo**: https://github.com/mylzwq/PPORank  
**Goal**: Full reproducibility challenge — reproduce all experiments reported in the paper across all datasets and analysis scenarios.

---

## Initial state of the repo

The initial commit (`a99850b`, 2021-05-23) contained the original authors' code, with the following issues preventing execution:

- Deprecated NumPy types: `np.float`, `np.int` (removed in NumPy 1.24), `from _testcapi import raise_exception`
- `preprocess/preprocess_fts_cl_drug.py` imported `from Baseline.CaDRRes import predict_CaDRRes` (module not present in the repo)
- All data paths were hardcoded without the `data/` subfolder (e.g. `"GDSC_ALL"` instead of `"data/GDSC_ALL"`)
- `load_dataset.py`: always re-downloaded data without checking if files were already present; WES/CNV/MET were read unconditionally (crash if absent); duplicated line `IC50_cell_ids, IC50_cell_names = [], []`; typo in docstring (`Parameterst`)
- `prepare.py`: wrong data path; loaded `gdsc_697_genes.csv` without checking existence; referenced `ess_genes_list` even when file was missing
- `main.py`: wrote results to `./results/{Data}/` without creating the directory; `SummaryWriter` commented out
- `prepare_loader.py`: unused import `from pygments.lexers import r`
- No `requirements.txt`; no `.gitignore`

---

## Changes made (relative to initial commit)

### Commit `d53e58b` — add gitignore
- Added `.gitignore`

### Commit `39d26cd` — path & minor fixes
Path corrections and minor bug fixes:

| File | Change |
|------|--------|
| `arguments.py` | `default="GDSC_ALL"` → `"data/GDSC_ALL"` |
| `prepare.py` | `default="GDSC_ALL"` → `"data/GDSC_ALL"` |
| `preprocess/prepare_simu.py` | `default="SimuData"` → `"data/SimuData"` |
| `preprocess/toxic_data.py` | GDSC_ALL paths → `data/GDSC_ALL` |
| `process_drug.py` | GDSC_ALL path → `data/GDSC_ALL` |
| `results_TCGA.py` | GDSC_ALL paths → `data/GDSC_ALL`; `np.int` → `int` |
| `utils.py` | Removed `from _testcapi import raise_exception`; replaced with `raise ValueError(...)`; changed `data_name == "GDSC_ALL"` → `"GDSC_ALL" in data_name` (for compatibility with `data/GDSC_ALL` path) |
| `prepare_loader.py` | Removed unused `from pygments.lexers import r` |

### Commit `765a810` — fix(GDSC preprocessing): fixed load_dataset and prepare.py

**`preprocess/load_dataset.py`** — full rewrite:
- Fixed import order (os/sys first, then third-party libraries)
- Added check for `Cell_line_RMA_proc_basalExp.txt` before downloading: skips download if already present
- Corrected data path: `os.getcwd()+"/data/GDSC_ALL"`
- `np.float` → `float`, `np.int` → `int`
- WES, CNV, MET made **optional**: if file is missing, logs `logger.warning` and continues instead of crashing
- Auxiliary variables initialized to `None` before conditional blocks (`WES_cell_ids`, `WES_CG`, `CNV_cell_ids`, `CNV_cna`, `MET_cell_ids`, `MET_met`) to avoid linter warnings
- Removed duplicated line `IC50_cell_ids, IC50_cell_names = [], []`
- Fixed docstring typo `Parameterst` → `Parameters`
- WES/CNV/MET saving conditioned on `if WES/CNV/MET is not None:`

**`prepare.py`**:
- Corrected `data_dir` path
- Loading of `gdsc_697_genes.csv` made optional: if file is missing, uses all genes
- Fixed bug: `ess_genes_list` was referenced even in the `None` case

**`preprocess/preprocess_fts_cl_drug.py`**:
- Removed `from Baseline.CaDRRes import predict_CaDRRes` (module absent)
- Inlined `predict_CaDRRes` function directly in the file
- GDSC_ALL and CCLE paths updated to `data/GDSC_ALL` and `data/CCLE`

### Commit `5f4506f` — update code
- `main.py`: re-enabled `SummaryWriter`; added automatic creation of `./results/{Data}/` directory before opening results file

### Commit `a738f83` — add misc.py
- `misc.py` added to project root: downloaded from KRL repo and adapted for Python 3 (was using Python 2 `print >>` syntax). Required by `results_TCGA.py`.

### Commit `bb55463` — fix: added logs and .sh pipeline (Ma7i5ab3, 2026-05-29)

| File | Change |
|------|--------|
| `run_pipeline.sh` | New file (119 lines): full bash pipeline script for GDSC — runs preprocessing → CV split → training with configurable `NUM_PROCESSES`, `F`, `ALGO`, `NFOLDS`, `CUDA_ID`; skips steps already done |
| `arguments.py` | Added `--shared_params` flag; added `--no_cuda` flag; improved device selection: `args.device = cuda:{cuda_id}` if available, else CPU |
| `main.py` | Added per-epoch timing and ETA in log; device info logged at startup (GPU name, CUDA version, memory); summary log at end with `best_ndcg` and total elapsed |
| `prepare.py` | Replaced bare `print` with structured `logger` calls; added per-fold timing; added pipeline start/end timing; added MF pretraining timing |
| `preprocess/load_dataset.py` | Replaced all `print` with `logger`; added step labels `[Step 1/5]`…`[Step 5/5]`; added per-step timing |

### Commit `55b0b27` — minor fixes (Ma7i5ab3, 2026-06-05)

| File | Change |
|------|--------|
| `.gitignore` | Added `Saved/*`, `results/*`, `runs/*` |
| `models/Policy.py` | `get_log_prob(scores, filter_masks, actions)` → `get_log_prob(scores, filter_masks, actions.squeeze(-1))` — fix tensor dimension mismatch |
| `set_log.py` | `args.Data` → `args.Data.replace('/', '_')` in log name construction — prevents `data/GDSC_ALL` from creating a subdirectory in the log name |
| `utils.py` | Same `.replace('/', '_')` fix in `create_model_name` for both PPO and non-PPO branches |

---

## Added `requirements.txt`
File created with pipeline dependencies:
```
torch>=1.12.0, numpy>=1.24.0, pandas>=1.3.0, scipy>=1.7.0,
scikit-learn>=1.0.0, matplotlib>=3.4.0, seaborn>=0.11.0,
tqdm>=4.62.0, PyYAML>=5.4.0, openpyxl>=3.0.0,
tensorboard>=2.8.0, h5py>=3.1.0, joblib>=1.0.0, urllib3>=1.26.0
```
Optional (only for `process_drug.py`): `pubchempy`, `rdkit`

---

## Data: current state

### `data/GDSC_ALL/`

| File | Source | Status | Notes |
|------|--------|--------|-------|
| `Cell_line_RMA_proc_basalExp.txt` | cancerrxgene.org (via Wayback Machine) | ✅ present | GEX: 1018 cell lines × 17737 genes |
| `TableS4A.xlsx` | cancerrxgene.org (via Wayback Machine) | ✅ present | IC50: 990 cell lines × 265 drugs |
| `TableS5C.xlsx` | cancerrxgene.org (via Wayback Machine) | ✅ present | Thresholds for IC50 normalization |
| `TableS1F.xlsx` | cancerrxgene.org (via Wayback Machine) | ✅ present | Drug IDs for matching |
| `CellLines_CG_BEMs/PANCAN_SEQ_BEM.txt` | cancerrxgene.org (via Wayback Machine) | ✅ present | WES: 961 cell lines × 300 genes |
| `CellLine_CNV_BEMs/PANCAN_CNA_BEM.rdata.txt` | cancerrxgene.org (via Wayback Machine) | ✅ present | CNV: 996 cell lines × 425 segments |
| `METH_CELLLINES_BEMs/PANCAN.txt` | — | ❌ unavailable | MET: permanently lost (410 Gone on cancerrxgene.org; not found on Wayback Machine, DepMap, Zenodo) |
| `gdsc_697_genes.csv` | CaDRReS GitHub (`CSB5/CaDRReS`) | ✅ created | 1856 essential genes from `essential_genes.txt`; used by `prepare.py` to filter GEX before computing the Pearson correlation kernel |
| `GDSC_GEX.npz` | generated by `load_dataset.py` | ✅ generated | 962 cell lines, 17737 genes, 265 drugs |
| `GDSC_WES.npz` | generated by `load_dataset.py` | ✅ generated | 953 cell lines, 300 genes, 265 drugs |
| `GDSC_CNV.npz` | generated by `load_dataset.py` | ✅ generated | 985 cell lines, 425 genes, 265 drugs |

### `data/CCLE/`

Collected proactively (not required by the main config `configG_FULL.yaml`, which uses only GDSC GEX):

| File | Source | Notes |
|------|--------|-------|
| `CCLE_all_abs_ic50_bayesian_sigmoid.csv` | CaDRReS GitHub (`input/`) | CCLE IC50 (504×24) |
| `CCLE_cellline_pcor_ess_genes.csv` | CaDRReS GitHub (`input/`) | Pre-computed Pearson kernel (1037×1037) |
| `CCLE_drugMedianGE0.txt` | CaDRReS GitHub (`misc/`) | List of 19 drugs with median IC50 ≥ 0 |

---

## Experiments to reproduce

Methods compared throughout: **EN**, **KRR**, **KRL**, **CaDRRes**, **PPO-w/o** (no eval signal), **PPORank**.  
Metrics: full-rank NDCG, NDCG@k and Precision@k for k ∈ {1, 5, 10}.

### Experiment 1 — Full dataset prediction (Section 3.1.4, Figure 3)
- **GDSC** (GEX only, 5-fold CV): compare mean NDCG across all methods.  
  Paper result: PPORank best (consistent improvement, SD=0.003); CaDRRes second (SD=0.011).
- **CCLE** (GEX only, 5-fold CV): compare mean NDCG.  
  Paper result: PPORank marginally better (0.7611 vs 0.7468 EN, 0.7432 CaDRRes).

### Experiment 2 — DNN vs PPORank over training epochs (Section 3.1.4, Figure 4)
- GDSC only: track NDCG over epochs (up to 600); PPORank approaches DNN trained with approximate NDCG loss.

### Experiment 3 — Multi-omic features (Section 3.1.5, Figure 5)
- GDSC with three feature combinations: GEX+WES, GEX+CNV, GEX+MET.  
  Paper result: GEX alone most informative; slight improvement with MET; CNV/WES negligible.  
  **⚠️ MET unavailable** — GEX+MET experiment cannot be reproduced.

### Experiment 4 — Sequential / online learning (Section 3.2, Figure 6)
- GDSC: start with 50% of training data, add 20 cell lines at each time step; track NDCG on fixed test set.

### Experiment 5 — Top-k ranking (Section 3.3, Figure 7)
- GDSC and CCLE: NDCG@k for k ∈ {1, 5, 10}, 3-fold CV (note: different from 5-fold used in Exp 1).

### Experiment 6 — External validation on TCGA BRCA (Section 3.4, Table 1)
- Train on GDSC; apply to TCGA BRCA cohort (1080 patients, harmonized gene expression).
- Evaluate lapatinib vs 4 PARP inhibitors (veliparib, olaparib, talazoparib, rucaparib) for HER2+ (n=163) and mBRCA TNBC (n=9 used in table, n=37 in text) patients.  
  Paper result: lapatinib ranked above all PARPi in 92% of HER2+ patients, only 22% of mBRCA TNBC.

### Experiment 7 — Primary simulations (Section 3.5.1, Table 2)
- 3 scenarios: linear (Y=0.2·XW+ε), cubic (Y=0.15·X³W+0.15·XW+ε), exponential (Y=0.1·exp(X)W+0.1·X³W+ε).
- N=1000 cell lines, 100 drugs, 2000 features; repeated at n=10000.
- Original cell line features X used directly (not Pearson kernel).

### Experiment 8 — Secondary simulations (Section 3.5.2, Table 3)
- Mixed scenario (40 drugs: first 20 linear, next 10 cubic, last 10 exponential), n=1000 and n=10000.

---

## Data status

| Dataset / Modality | Status | Notes |
|--------------------|--------|-------|
| GDSC GEX | ✅ `GDSC_GEX.npz` generated | 962 cell lines, 17737 genes, 265 drugs |
| GDSC WES | ✅ `GDSC_WES.npz` generated | 953 cell lines, 300 genes |
| GDSC CNV | ✅ `GDSC_CNV.npz` generated | 985 cell lines, 425 segments |
| GDSC MET | ❌ permanently lost | File not found on cancerrxgene.org, Wayback Machine, DepMap, Zenodo |
| CCLE | ✅ kernel + IC50 present | `CCLE_cellline_pcor_ess_genes.csv` (1037×1037), `CCLE_all_abs_ic50_bayesian_sigmoid.csv` (504×24), `CCLE_drugMedianGE0.txt` |
| SimuData | ❓ not generated yet | `preprocess/prepare_simu.py` generates it; needs to be run |
| TCGA BRCA | ⚠️ partial | `misc.py` present (commit `a738f83`); `TCGA_BRCA.npz` and `TCGA_BRCA_clinical.csv.gz` still missing — require R pipeline (see TCGA section below) |

**Note on drug count discrepancy**: the paper reports 223 GDSC drugs (after removing toxic drugs) and 19 CCLE drugs, but the config uses `k_max: [265, 26]` and the .npz files contain 265 GDSC drugs. Toxic drug filtering appears to happen inside the pipeline (possibly in `preprocess/toxic_data.py`) and must be verified.

---

## Execution pipeline

### GDSC (Experiments 1, 2, 3, 4, 5)

```bash
conda activate pporank

# Step 1 — generate .npz from raw data (already done)
python preprocess/load_dataset.py

# Step 2 — generate CV fold splits
python prepare.py --config configs/configG_FULL.yaml

# Step 3 — training
python main.py --config configs/configG_FULL.yaml
```

### CCLE (Experiments 1, 5)

```bash
# Step 1 — preprocess CCLE features
python preprocess/preprocess_fts_cl_drug.py --Data data/CCLE

# Step 2 — generate CV fold splits (config to be created)
python prepare.py --config configs/configC.yaml

# Step 3 — training
python main.py --config configs/configC.yaml
```

### Simulations (Experiments 7, 8)

```bash
python preprocess/prepare_simu.py   # generate synthetic data
python prepare.py --config configs/configSimu.yaml   # config to be created
python main.py --config configs/configSimu.yaml
```

### TCGA validation (Experiment 6)

```bash
# Requires TCGA_BRCA.npz and TCGA_BRCA_clinical.csv.gz — see TCGA data pipeline below
python results_TCGA.py
```

---

## TCGA data pipeline (Experiment 6)

`results_TCGA.py` needs two files in `data/GDSC_ALL/`:

| File | Content |
|------|---------|
| `TCGA_BRCA.npz` | `test_X` (harmonized TCGA gene expression, 1080 patients) and `test_ids` (TCGA patient IDs) |
| `TCGA_BRCA_clinical.csv.gz` | Clinical annotations indexed by patient ID: columns ER, PR, HER2, CHR17, BRCA_germline, JAK2_RPPA |

Also requires `misc.py` in the project root — **downloaded from KRL repo and adapted for Python 3** (was using Python 2 `print >>` syntax).

### Step 1 — Generate harmonized TCGA gene expression (R)

The paper uses the pRRophetic Bioconductor package (Geeleher et al 2017, Genome Research) to harmonize TCGA RNA-seq (Level 3, Illumina HiSeq v2) with GDSC microarray gene expression.

```r
# Install pRRophetic in R
if (!require("BiocManager")) install.packages("BiocManager")
BiocManager::install("pRRophetic")

# Download TCGA BRCA RNA-seq Level 3 from Broad Institute Firehose:
# https://gdac.broadinstitute.org/ → BRCA → Stddata 2016-01-28 → mRNAseq_Preprocess

# Run harmonization (produces pRRophetic_TCGA_BRCA_trainFrame.csv.gz,
#                             pRRophetic_TCGA_BRCA_testFrame.csv.gz,
#                             pRRophetic_TCGA_BRCA_preds.csv.gz)
# See: Geeleher et al. Genome Res. 2017;27(10):1743-1751
```

### Step 2 — Create TCGA_BRCA.npz (Python, adapted from KRL)

The KRL repo provides `create_TCGA_data.py` which reads the pRRophetic output and `GDSC_GEX.npz`, then saves `KRL_data_for_TCGA_BRCA.npz` with keys `train_X`, `train_Y`, `test_X`, `test_ids`. The PPORank `results_TCGA.py` reads `TCGA_BRCA.npz` with the same keys, so the KRL output can be used directly after renaming/copying.

Note: `create_TCGA_data.py` from KRL also uses Python 2 syntax and would need adaptation.

### Step 3 — Assemble TCGA_BRCA_clinical.csv.gz

Required columns and sources:
| Column | Source |
|--------|--------|
| `HER2`, `ER`, `PR` | TCGA BRCA clinical annotations (GDC or cBioPortal) |
| `CHR17` | Chromosome 17 copy number from TCGA CNV data |
| `BRCA_germline` | Maxwell et al. Nat Commun 2017 — germline BRCA1/2 loss-of-function mutations |
| `JAK2_RPPA` | TCGA RPPA (reverse-phase protein array) data |

---

## Open issues / to verify

| Issue | Priority | Notes |
|-------|----------|-------|
| **MET modality** | Low | Cannot reproduce GEX+MET experiment (Figure 5 incomplete) |
| **TCGA BRCA data** | High | `TCGA_BRCA.npz` not present; must download from Broad Institute Firehose 2016-01-28 and harmonize with GDSC gene expression (pipeline from Geeleher et al 2017) |
| **Toxic drug filtering** | High | Paper uses 223 GDSC drugs / 19 CCLE drugs; `.npz` files have 265 / 24; verify that `preprocess/toxic_data.py` is called correctly and produces the right filtered dataset |
| **GDSC vs GDSC_ALL** | Medium | `config.yaml` uses `data: GDSC`; `configG_FULL.yaml` uses `data: GDSC_ALL`; check if they refer to the same preprocessed data or to two different subsets |
| **CCLE config** | Medium | No `configC.yaml` exists; need to create it with correct dataset path, methods, and hyperparameters |
| **SimuData config** | Medium | No config for simulation experiments; need to create it |
| **CaDRRes preprocessing** | High | `preprocess_fts_cl_drug.py` must be run before training to generate CaDRRes input features; not yet verified |
| **`prepare.py` not yet run** | High | Must run before any training; no fold splits found in `data/GDSC_ALL/CV/` as of 2026-06-05 |
