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

### Session 2026-06-09 — TCGA pipeline + SimuData

#### New scripts created

| File | Purpose |
|------|---------|
| `preprocess/breast_cancer_analysis.R` | pRRophetic ComBat harmonization: TCGA RNA-seq + GDSC microarray → 3 CSV.gz intermediate files |
| `preprocess/create_TCGA_data.py` | Python 3 port of KRL `create_TCGA_data.py` → `TCGA_BRCA.npz` |
| `preprocess/create_TCGA_clinical.py` | Assembles `TCGA_BRCA_clinical.csv.gz` from 4 sources |
| `preprocess/generate_simu_data.py` | Generates X, W, Y CSVs for simulation experiments (linear/quad/exp scenarios) |
| `preprocess/split_simu_data.py` | Standalone CV-fold splitter for SimuData (no torch/CaDRRes dependency) |
| `Simulation/__init__.py` | Package init |
| `Simulation/utils_simu.py` | `noramlize_y` and `read_simu` stubs (required by `prepare_simu.py`) |
| `preprocess/__init__.py` | Added to make preprocess a proper package |

#### Bugs fixed

| Script | Bug | Fix |
|--------|-----|-----|
| `breast_cancer_analysis.R` | Pattern `RSEM_genes_normalized` not present; actual file is `BRCA.uncv2.mRNAseq_RSEM_normalized_log2.txt` | Added `mRNAseq_RSEM_normalized_log2` as first pattern candidate |
| `breast_cancer_analysis.R` | pRRophetic internal GDSC has 46 NA-named + 4 duplicate cell line columns → ComBat crash | Remove bad columns before `homogenizeData()` |
| `create_TCGA_data.py` | Used `gdsc_gex['cell_ids']` (numeric strings) instead of `cell_names` → 0 intersection | Changed to `cell_names` |
| `create_TCGA_data.py` | TCGA barcodes from pRRophetic are short (`TCGA-XX-XXXX-01`, 15 chars); filter `x[13:16]=='01A'` failed | Changed to `x[13:15]=='01'` |
| `create_TCGA_data.py` | NaN filter `~np.isnan(test_X[0])` only checked first patient | Changed to `~np.any(np.isnan(test_X), axis=0)` → removes 4276 genes with any NaN |
| `create_TCGA_clinical.py` | `Clinical_Pick_Tier1` file has no ER/PR/HER2 IHC columns | Switched to `Merge_Clinical.Level_1/BRCA.clin.merged.txt` |
| `create_TCGA_clinical.py` | Glob for SNP6 segment dir returned `.tar.gz` first | Simplified: search for `.seg.txt` recursively from data_dir |

#### Data downloaded

- Firehose BRCA 2016-01-28: mRNAseq (1.5GB), Merge_Clinical (3.5MB), RPPA (2MB), SNP6 hg19 (18MB)
- pRRophetic_0.5.tar.gz from OSF (493MB) — installed via `R CMD INSTALL`
- Maxwell et al. 2017 Supp Data 1 (`MOESM2_ESM.xlsx`) from Springer CDN

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

**Method sets differ by experiment** (verified against paper text, 2026-06-13):
- **Exp 1 / 3 / 5 (real data, Fig 3/5/7)**: EN, KRR, **SRMF**, CaDRReS, **KRL**, PPORank.
  *No* PPO-w/o here.
- **Exp 7 / 8 (simulations, Table 2/3)**: EN, KRR, KRL, CaDRReS, **PPO-w/o**, PPORank.
  *No* SRMF (needs drug similarity).

**PPO-w/o** = PPORank trained **without positive evaluation signals**, defined only for the
simulations (§3.5.1): during training, placing a *non-sensitive* drug in a ranking slot while
sensitive candidates remain = **negative** signal, otherwise **positive** (sensitive = response
above the per-drug median). PPO-w/o drops the positive component. **It is a reward ablation,
not the `--pretrain` flag.**

**KRL** (§3.1.2): RBF kernel; input features are the **RMA basal expression of all 17 737 genes**
(not the 1610-essential-gene Pearson kernel used by PPO/EN). λ, γ tuned over grids
(`krl_lambdas`/`krl_gammas` in configs). Not implemented in the released code — must be written
like EN/KRR in `baselines.py`. Ref: He, Folkman, Borgwardt, *Bioinformatics* 2018;34(16):2808.

Metrics: full-rank NDCG, NDCG@k and Precision@k for k ∈ {1, 5, 10}.

### Experiment 1 — Full dataset prediction (Section 3.1.4, Figure 3)
- **GDSC** (GEX only, 5-fold CV): compare mean NDCG across all methods.  
  Paper result: PPORank best (consistent improvement, SD=0.003); CaDRRes second (SD=0.011).
- **CCLE** (GEX only, 5-fold CV): compare mean NDCG.  
  Paper result: PPORank marginally better (0.7611 vs 0.7468 EN, 0.7432 CaDRRes).
- **EN detail** (§3.1.2): one model per drug, scikit-learn, **l1_ratio fixed = 0.5** (our
  `baselines.py` tunes l1_ratio over a grid → minor deviation to note).
- **SRMF caveat**: paper lists it among Fig 3 methods but also states it "cannot make
  predictions on unseen cell lines"; how it was scored under unseen-cell-line 5-fold CV is
  unclear — to verify before attempting.

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
- Methods: EN, KRR, KRL, CaDRReS, **PPO-w/o**, PPORank (no SRMF — needs drug similarity).
- Preprocessing: negative responses set missing; 50% of remaining set missing; sensitive =
  above per-drug median (drives the positive/negative evaluation signal for PPO vs PPO-w/o).
- Paper Table 2 (mean NDCG, n=1000 / n=10000): PPORank 0.790/0.832 (sc1), 0.651/0.705 (sc2),
  0.941/0.959 (sc3); PPO-w/o 0.798/0.811, 0.670/0.693, 0.940/0.948.

### Experiment 8 — Secondary simulations (Section 3.5.2, Table 3)
- Mixed scenario (40 drugs: first 20 linear, next 10 cubic, last 10 exponential), n=1000 and n=10000.
- Same method set as Exp 7. Paper Table 3 (mean NDCG): PPORank 0.721/0.732, PPO-w/o 0.701/0.713,
  KRL 0.689/0.707, CaDRReS 0.647/0.689, EN 0.645/0.679, KRR 0.641/0.650.

---

## Data status

| Dataset / Modality | Status | Notes |
|--------------------|--------|-------|
| GDSC GEX | ✅ `GDSC_GEX.npz` generated | 962 cell lines, 17737 genes, 265 drugs |
| GDSC WES | ✅ `GDSC_WES.npz` generated | 953 cell lines, 300 genes |
| GDSC CNV | ✅ `GDSC_CNV.npz` generated | 985 cell lines, 425 segments |
| GDSC MET | ❌ permanently lost | File not found on cancerrxgene.org, Wayback Machine, DepMap, Zenodo |
| CCLE | ✅ kernel + IC50 present | `CCLE_cellline_pcor_ess_genes.csv` (1037×1037), `CCLE_all_abs_ic50_bayesian_sigmoid.csv` (504×24), `CCLE_drugMedianGE0.txt` |
| SimuData | ✅ generated | N=1000 and N=10000; 3 scenarios (linear/quad/exp); 5-fold CV splits in `data/SimuData/N*/`; see TCGA section for details |
| TCGA BRCA | ✅ complete | `TCGA_BRCA.npz` (960 CCL × 10915 genes, 1093 patients) and `TCGA_BRCA_clinical.csv.gz` (2245 patients × 6 cols) generated 2026-06-09 |

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
# Data already generated (2026-06-09):
#   python preprocess/generate_simu_data.py --N 1000
#   python preprocess/generate_simu_data.py --N 10000
#   python preprocess/split_simu_data.py --N 1000
#   python preprocess/split_simu_data.py --N 10000

python prepare.py --config configs/configSimu.yaml   # config to be created
python main.py --config configs/configSimu.yaml
```

### TCGA validation (Experiment 6)

```bash
# Requires TCGA_BRCA.npz and TCGA_BRCA_clinical.csv.gz — see TCGA data pipeline below
python results_TCGA.py
```

---

## TCGA data pipeline (Experiment 6) — COMPLETED 2026-06-09

Both output files are present in `data/GDSC_ALL/`. This section documents the full pipeline for reproducibility.

### Output files

| File | Content |
|------|---------|
| `TCGA_BRCA.npz` | `train_X` (960×10915), `train_Y` (960×265), `test_X` (1093×10915), `test_ids` (1093 patient IDs), `drug_ids`, `drug_names` |
| `TCGA_BRCA_clinical.csv.gz` | 2245 patients × 6 columns: ER, PR, HER2, CHR17, BRCA_germline, JAK2_RPPA |

Sanity check vs paper: HER2+ = 164 (paper: 163 ✓), BRCA_germline ∩ test_ids = 37 (paper: 37 ✓).

### Step 0 — Install pRRophetic

pRRophetic is no longer on CRAN or Bioconductor. Install from the OSF archive:

```bash
# Install R dependencies
Rscript -e 'install.packages(c("car","ridge"), repos="https://cloud.r-project.org")'
Rscript -e 'BiocManager::install(c("preprocessCore","genefilter","sva"))'

# Download tarball
wget -O pRRophetic_0.5.tar.gz "https://osf.io/dwzce/?action=download"
R CMD INSTALL pRRophetic_0.5.tar.gz
```

### Step 1 — Download Firehose archives

```bash
cd data/GDSC_ALL
BASE="https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/BRCA/20160128"
curl -L -O "$BASE/gdac.broadinstitute.org_BRCA.mRNAseq_Preprocess.Level_3.2016012800.0.0.tar.gz"
curl -L -O "$BASE/gdac.broadinstitute.org_BRCA.Merge_Clinical.Level_1.2016012800.0.0.tar.gz"
curl -L -O "$BASE/gdac.broadinstitute.org_BRCA.RPPA_AnnotateWithGene.Level_3.2016012800.0.0.tar.gz"
curl -L -O "$BASE/gdac.broadinstitute.org_BRCA.Merge_snp__genome_wide_snp_6__broad_mit_edu__Level_3__segmented_scna_hg19__seg.Level_3.2016012800.0.0.tar.gz"
for f in *.tar.gz; do tar -xzf "$f"; done
```

Maxwell et al. 2017 Supplementary Data 1 (germline BRCA mutations) can be auto-downloaded:
```python
# Already done — data/GDSC_ALL/maxwell2017_supp_data1.xlsx
# Source: https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-017-00388-9/MediaObjects/41467_2017_388_MOESM2_ESM.xlsx
```

### Step 2 — pRRophetic ComBat harmonization

```bash
Rscript preprocess/breast_cancer_analysis.R
```

Produces `pRRophetic_TCGA_BRCA_{trainFrame,testFrame,preds}.csv.gz` in `data/GDSC_ALL/`.

Key fixes applied vs naive pRRophetic call:
- Expression file: `BRCA.uncv2.mRNAseq_RSEM_normalized_log2.txt` (1212 samples, already log2) — not RPKM
- GDSC internal data has 46 NA-named + 4 duplicate cell line columns → removed before ComBat
- 15097 common genes; ComBat reports 795643 missing values (handled internally)

### Step 3 — Assemble TCGA_BRCA.npz

```bash
python preprocess/create_TCGA_data.py
```

Key fixes vs original KRL Python 2 code:
- `gdsc_gex['cell_names']` (not `cell_ids`) for cell line name matching → 961 matches
- TCGA barcodes from pRRophetic are short (15 chars, end in `01` not `01A`) → filter `x[13:15] == '01'`
- NaN filter: `~np.any(np.isnan(test_X), axis=0)` (any patient, not just first) → 10915 genes retained

### Step 4 — Assemble TCGA_BRCA_clinical.csv.gz

```bash
python preprocess/create_TCGA_clinical.py   # also: pip install openpyxl
```

Sources:
| Column | Source file |
|--------|-------------|
| ER, PR, HER2 | `Merge_Clinical.Level_1/BRCA.clin.merged.txt` (columns: `patient.breast_carcinoma_estrogen_receptor_status`, `patient.breast_carcinoma_progesterone_receptor_status`, `patient.lab_proc_her2_neu_immunohistochemistry_receptor_status`) |
| CHR17 | SNP6 hg19 segments — weighted mean Segment_Mean over chr17 |
| BRCA_germline | `maxwell2017_supp_data1.xlsx` MOESM2 |
| JAK2_RPPA | `RPPA_AnnotateWithGene.Level_3/BRCA.rppa.txt` |

Note: CHR17 weighted mean across full chromosome 17 is always < 2 (chr17_pos = 0 for all patients). This is expected: the ≥2 threshold targets focal HER2 amplicon segments, not the chromosome-wide average. With chr17_pos=False for all, TNBC is defined purely by HER2/ER/PR IHC.

---

## Performance optimization & training robustness (2026-06-10 → 06-11)

Commits `2dc0f71` → `9b2b857`. After the full data pipeline was working, the
bottleneck became runtime: MF (CaDRRes) pretraining and PPO training were too
slow to iterate on. This block speeds up both, adds a fast sanity-check
pipeline, and makes PPO training stop early when it stops improving.

### Quick pipeline (`0bcdb96`)

New fast/dev variant that runs the whole pipeline end-to-end in minutes for
sanity checks — **not for reported results**.

- `run_pipeline_quick.sh` + `configs/configG_FULL_quick.yaml`
- Writes all artifacts under a separate `data/GDSC_ALL_QUICK/` tree, so it never
  touches the real `data/GDSC_ALL` folds, models, or results.
- Reduced: `nfolds 5→2`, MF `iters 20000→1000`, PPO `epochs 1400→30`, training
  cell-line fraction `prop 1.0→0.5`. Keeps `f=100` and `num_processes=32`.
- Constraint: keep `f ≥ deep_out_size + 4` (see the derivation section below). The
  original "f ≥ 68" note assumed `deep_out_size=64`; with `deep_out_size=32` it is
  `f ≥ 36`. (The earlier mention of `sample_concatenate`'s reshape was stale — the
  `obs_critic` path there is commented out; the real constraint is in `ConvValueNet`.)

### MF pretraining speedup — `Response_decompose` (`d32f99b`, `2dc0f71`)

`preprocess/preprocess_fts_cl_drug.py`:
- Since `Y = I_M`, the prediction collapses `X @ WP @ WQ.T @ Y` → `X @ WP @ WQ.T`,
  and the WP/WQ gradient updates become two matmuls each instead of the original
  3-D broadcast (`WP` update was ~2.6 GB, `WQ` ~432 MB of temporaries — now gone).
- `WE` is all-ones, so the `multiply(..., WE)` masks were dropped as no-ops.
- Removed the per-epoch best-error rollback (saving/restoring `WP/WQ/mu/b_p/b_q`
  every iteration); NDCG and checkpointing now computed only every 100 epochs.
- Added timestamped progress logging with per-block time, elapsed, and ETA.

`prepare.py` (`Pretrained_MF_split`):
- New `--iters` CLI arg (default 20000) instead of the hardcoded default.
- **Skip-if-done**: a fold whose `…/{f}Dim/WPmatrix.csv` already exists is
  skipped, so an interrupted run resumes without redoing finished folds.
- Timestamped per-fold logging.

`preprocess/pp_gene_original.py`: vectorized the Pearson similarity matrix
(z-score columns then `Zᵀ @ Z / n_genes`) instead of the nested per-sample-pair
`stats.pearsonr` loop.

### PPO forward-pass speedup (`c2c70c6`, `a14ea3e`, `4596e3d`)

`models/Policy.py`:
- `Deep_Cross_Policy.forward`: cell features are identical for all M drugs of a
  cell, so `cell_emb` is now computed once and `expand`-ed over M instead of
  building the full `(B,M,cell_dim)` tensor. Removed the `input.cpu()` / `.clone()`
  round-trips and now respects the input's device.
- New `forward_from_obs(obs_actor, drug_inds)`: a lean path for
  `evaluate_actions` that skips reconstructing the `(B,M,P+1)` input tensor.
- `PPO_Policy.evaluate_actions`: builds `drug_inds` with `torch.arange(...).expand`
  on-device instead of round-tripping through numpy.

`Agent/storage.py`:
- Replaced Python-list path accumulation (`log_pi`, `dist_entropy`, `value_pred`)
  with tensor slices / `expand`, and the downstream stacking with `torch.cat`.
- **Vectorized GAE**: the reverse per-step Python loop is replaced by a
  reverse-cumsum with discount factors, computed on CPU per cell to avoid
  M individual CUDA sync points.

### Training-loop fixes & robustness (`dff663d`, `d9c5f10`, `c9aed1e`, `9b2b857`)

`main.py`:
- `DataLoader(num_workers=4 → 0)` (workers were hurting, not helping here);
  `pin_memory=True` kept.
- `--resume` fixed: `torch.load(..., map_location=device)`, and load the merged
  `actor_critic` via `checkpoint['Policy_state_dict']` (the old code referenced
  non-existent `policy_state_dict`/`value_state_dict` keys).
- Checkpoint-save guard bug fixed: `epoch == num_updates - 1` → `epoch == epochs - 1`.
- **Early stopping**: new `--early_stopping_patience` arg (default 50, `0` =
  disabled). Counter resets on each new best test NDCG; training breaks after
  `patience` epochs without improvement.

---

## Results aggregation fixed + first GDSC PPO run (2026-06-12)

First full GDSC run on the H100 server completed all 5 PPO folds (early stopping
hit each fold), but **STEP 4 (aggregation) crashed**:
`FileNotFoundError: ./configs/configS_base.yaml`.

### Bugs fixed in `results.py`

- `__main__` ignored its CLI args and called
  `save_exp_baselines_results("./configs/configS_base.yaml")` (a leftover
  SimuData path that doesn't exist), while `run_pipeline.sh` was passing
  `--config`. Wired `__main__` to the existing `parse_args()` →
  `save_exp_baselines_results(args.config, single_k=args.k)`.
- Dataset-kind detection used `data_name.startswith("GDSC")` / `== "CCLE"` /
  `== "SimuData"`, which fails for a path-style id like `data/GDSC_ALL` (what
  training actually used). Now derives `data_kind = os.path.basename(data_name)`
  and branches on that, keeping the full `data_name` for result paths.
- `N`/`P`/`M` read via `config.get(...)` instead of hard indexing (they only
  feed the `data_form` string, which is unused for FULL analysis).

### New aggregation config + pipeline wiring

- `configs/configG_FULL_ppo.yaml` (new): aggregation config targeting the
  trained PPO output — `methods: ['ppo']`, `data: data/GDSC_ALL`, `f: 100`
  (matching `--Data`/`--f` used in STEP 3). The baseline `configG_FULL.yaml`
  is left untouched. Result path resolves to
  `results/data/GDSC_ALL/FULL/100Dim/ppo/ppo_{fold}.npz`.
- `run_pipeline.sh` STEP 4 now points at `configG_FULL_ppo.yaml`.

### First GDSC PPORank results (5-fold CV, FULL, f=100, essential genes)

Re-ran `run_pipeline.sh` — STEP 1–3 all skipped (artifacts present), STEP 4 ran
in 2s and wrote `results_ppo.txt`.

| Metric | Mean (5 folds) | Per-fold |
|--------|---------------|----------|
| NDCG@1 | 0.356 | 0.335 / 0.350 / 0.357 / 0.323 / 0.413 |
| NDCG@5 | 0.367 | 0.363 / 0.351 / 0.371 / 0.367 / 0.385 |
| NDCG@10 | 0.391 | 0.394 / 0.380 / 0.401 / 0.387 / 0.392 |
| NDCG@265 (full) | 0.717 | 0.721 / 0.708 / 0.715 / 0.716 / 0.724 |
| Precision@1 | 0.149 | — |
| Precision@5 | 0.190 | — |
| Precision@10 | 0.196 | — |

Sanity check: NDCG@265 matches the per-fold `best_ndcg` from the training logs
(0.7208 / 0.7076 / 0.7139 / 0.7146 / 0.7238), confirming aggregation reads the
correct `.npz`. `Precision@265 = 1.0` is trivial (k = total drug count).

**Note**: the per-fold `.npz` is overwritten on each new best epoch, so it holds
the **best-epoch** test predictions — consistent with early stopping. Baseline
methods (EN/KRR/KRL/CaDRRes) for the Experiment 1 comparison are **not yet run**;
this is PPO only.

---

## Toxic-drug filter — 223 vs 265 GDSC drugs (2026-06-12)

The first GDSC run trained on **265** drugs, but the paper reports **223** (after
removing toxic drugs). Investigated where the filter should happen and why it was
skipped.

### Root cause — text vs released code

- The paper (Section 3.1.1) describes the filter as a *heuristic outlier detection*
  (Iorio et al. 2016, ref 59) on drug-specific sensitivity thresholds, **plus**
  exclusion of drugs without a PubChem ID (and 15 duplicate-PubChem drugs) → 223.
- The **released code** does *not* implement that. The 265→223 reduction lives only
  in `prepare.py`'s `Data_All=False` branch (lines 194–197), which reads CaDRReS'
  drug list `{Source}_drugMedianGE0.txt` (drugs with median sensitivity ≥ 0).
  Author's own comment at line 196: `# 265 drugs, after delete from median 1um, has 223 drugs`.
- Our config (`configG_FULL.yaml`) sets `Data_All: True`, whose branch loads
  `GDSC_GEX.npz` directly and takes all 265 `drug_ids` with **no drug filter** → 265.
- `preprocess/toxic_data.py` (the would-be Iorio-style filter) is **incomplete and
  unrunnable**: it reads a missing `data/GDSC_ALL/toxic_drug_ids.csv`, has a
  hardcoded author path (`/home/liux3941/RL/RL_GDSC/GDSC_ALL/gdsc_drugMedianGE0.txt`),
  and writes no output (truncated).

### Decision — use CaDRReS' list (reproduce the code, not re-derive)

For a faithful reproduction we replicate what the authors' pipeline actually does
(median-GE0 list), rather than re-implementing Iorio's heuristic (which would risk a
*different* 223 set and add a confound). **Caveat to report**: the criterion
described in the paper (Iorio + PubChem) and the one implemented (CaDRReS median-GE0)
both yield 223 but are not formally verified to be the identical drug set.

### Actions taken

- Recovered `gdsc_drugMedianGE0.txt` from CaDRReS GitHub
  (`CSB5/CaDRReS/misc/gdsc_drugMedianGE0.txt`, 223 drug IDs) → saved as
  `data/GDSC_ALL/GDSC_drugMedianGE0.txt`. Verified 223/223 IDs intersect the
  `.npz` `drug_ids` (zero missing).
- `prepare.py` (`Data_All=True` branch, after `Y`/`X` construction): added a guarded
  filter that reads `{Source}_drugMedianGE0.txt`, restricts `Y` to the selected
  drugs, and logs `Drug filter (...): 265 -> 223 drugs`. Falls back to a warning +
  all drugs if the list is absent (safe for CCLE/Simu).
- Re-ran the pipeline on 223 drugs. Step 2 log confirms `265 -> 223 drugs` and the
  1610 essential-gene intersection; folds rebuilt (train ~769 / test ~193).

**Note**: `data/` is gitignored, so `GDSC_drugMedianGE0.txt` does **not** sync via
git — it must be copied to each machine manually (or re-`curl`-ed from CaDRReS).

### Incident — truncated `GDSC_GEX.npz` on the server

While re-running, `np.load` crashed with `BadZipFile: File is not a zip file`. The
server's `GDSC_GEX.npz` was **truncated to 20 MB** (vs 140 MB; valid ZIP header,
correct `X` shape `(962, 17737)`, but cut off → missing central directory). Caused
by an interrupted write/transfer the day before (dated Jun 11 15:02), **not** by the
code or git (the file is gitignored and untracked). Fixed by re-`scp`-ing the valid
copy from the Mac and verifying `md5sum = fce179909a9b518a385e16ee619ebd4d`.

---

## Full 223-drug run + EN/KRR baselines (2026-06-12)

First end-to-end run on the **223-drug** set (toxic filter active) with the whole
pipeline succeeding, STEP 4 included. Total elapsed **28113s (~7.8h)** on the H100:
Step 1 skipped (`GDSC_GEX.npz` present), Step 2 (CV split + MF pretrain) 1208s, Step 3
(5-fold PPO) the bulk, Step 4 (aggregation) 2s → `results_ppo.txt`. EN/KRR baselines
ran in a separate `baselines.py` job (config `configG_FULL_compare.yaml`, val_frac 0.2).

### PPORank (5-fold CV, FULL, f=100, 223 drugs, essential genes)

| Metric | Mean (5 folds) | Per-fold |
|--------|---------------|----------|
| NDCG@1 | 0.356 | 0.331 / 0.348 / 0.318 / 0.368 / 0.415 |
| NDCG@5 | 0.356 | 0.356 / 0.347 / 0.328 / 0.378 / 0.372 |
| NDCG@10 | 0.389 | 0.394 / 0.388 / 0.361 / 0.409 / 0.392 |
| NDCG@full | 0.709 | 0.712 / 0.707 / 0.693 / 0.719 / 0.713 |
| Precision@1 | 0.145 | — |
| Precision@5 | 0.177 | — |
| Precision@10 | 0.194 | — |

Per-fold `best_ndcg` (training logs): 0.7127 / 0.7076 / 0.6940 / 0.7184 / 0.7122,
matching the aggregated full-rank NDCG (best-epoch predictions under early stopping).

### Baselines (same folds, f=100)

| Method | Test NDCG mean | Per-fold | Best hyperparams |
|--------|---------------|----------|------------------|
| **EN** | **0.778** | 0.783 / 0.778 / 0.770 / 0.786 / 0.774 | α=0.1, l1_ratio 0.5–0.9 |
| **KRR** | 0.649 | 0.648 / 0.640 / 0.654 / 0.650 / 0.653 | α=10–100 |

EN training is slow (~2650s/fold, grid over α × l1_ratio); KRR ~6s/fold.

### Key finding — PPO underperforms EN, RL phase stalls

**Ordering: EN (0.778) > PPO (0.709) > KRR (0.649).** PPO is ~0.07 NDCG below the
Elastic Net baseline. The training dynamics explain it: on **every** fold `test_ndcg`
peaks within the first ~5–20 epochs and then drifts down, so early stopping fires at
epoch 54–70 with the best checkpoint coming from very early (e.g. Fold 3 best at ep 8,
Fold 2 at ~ep 4). `loss` stays flat in the ~2.3–2.9e7 band throughout. → the PPO/RL
phase is **not improving over the MF (CaDRReS) warm-start** — at best holding, at worst
slowly eroding it. This contradicts the paper, where PPORank is the top method.

Candidate causes to investigate next: (1) advantage/reward scaling — rewards ~85k,
loss ~2e7, advantages may be unnormalized so gradient is dominated by scale; (2) actor
LR / entropy too high (peak-then-decay is classic over-stepping); (3) confirm against
the paper's reported PPORank-vs-EN gap (possible reproduction discrepancy).

---

## CaDRReS aggregated → 4-method Exp-1 table (2026-06-13)

Ran `python results.py --config configs/configG_FULL_compare.yaml` (no `--k`) on the
server. CaDRReS `.npz` (written by MF pretrain) aggregated alongside ppo/EN/KRR. Also
set `k_max: [223,26]` in the compare config so the full-rank label reads `rank_223`
(NDCG@223 == NDCG@265 numerically — DCG stops at the 223 available drugs). Full
4-method comparison, GDSC FULL, 5-fold CV, f=100, 223 drugs:

| Method | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@full | Prec@1 | Prec@5 | Prec@10 |
|--------|----:|----:|----:|----:|----:|----:|----:|
| **EN** | **0.437** | **0.500** | **0.529** | **0.778** | **0.201** | **0.300** | **0.313** |
| PPORank | 0.356 | 0.356 | 0.389 | 0.709 | 0.145 | 0.177 | 0.194 |
| CaDRReS | 0.247 | 0.282 | 0.308 | 0.668 | 0.073 | 0.118 | 0.140 |
| KRR | 0.218 | 0.239 | 0.267 | 0.649 | 0.068 | 0.096 | 0.126 |

### Revised diagnosis — PPO *does* beat CaDRReS; the anomaly is EN

Correcting the earlier "RL adds nothing" framing: **PPO (0.709) > its CaDRReS warm-start
(0.668)** by +0.041 full-rank (+0.11 at NDCG@1). The RL phase *does* contribute; it just
saturates early (peak ep ~15–20) and stays far below EN. So the open problem is two-fold:

- **(a) EN is anomalously strong** — dominates at *every* k, including NDCG@1 where a
  ranking method should win. Paper has EN ≈ CaDRReS ≈ PPORank (all close on CCLE); here
  EN is +0.07 over PPO. Suspect a possible advantage in our EN eval (per-drug regression
  on the same fold) — worth auditing `baselines.py` for any train/test leakage or a
  metric mismatch vs PPO's eval.
- **(b) Ordering vs paper**: paper GDSC = PPORank > CaDRReS > rest; ours = EN > PPORank >
  CaDRReS > KRR. PPORank drops to 2nd, CaDRReS to 3rd.

**CaDRReS dimension mismatch**: paper sets CaDRReS latent dim = **10**; our MF pretrain runs
at **f=100** (to match PPO's projection). This likely depresses our CaDRReS (3rd instead of
paper's 2nd) and is a known deviation to flag in the report.

Still missing for the full Fig-3 set: **KRL** and **SRMF**.

---

## Feature-representation bug found + fixed (2026-06-13)

Systematic paper-vs-run audit of Exp 1 surfaced a **critical deviation**: PPORank and the
CaDRReS warm-start were trained on the **raw 1610 essential-gene expression** (z-scored),
not the **Pearson cell-line similarity kernel** the paper/CaDRReS use (§3.1.1, ref 13).

### Root cause

`prepare.py` builds the Pearson kernel (`kernel_feature_df`, cell-line × cell-line) and
saves it per fold as `Xtrain_kernel.csv` / `Xtest_kernel.csv`, but the `Data_All=True`
branch writes the **raw essential-gene matrix** into `Xtrain_rawDf.csv` (line ~247 uses `X`,
which at line 172 is the gene-expression subset — not the kernel). The `Data_All=False`
branch instead sets `X` = the precomputed kernel (line 229), so the *same* `Xtrain_rawDf.csv`
holds different things in the two branches. Consumers:

- PPO → `utils.read_FULL` reads `Xtrain_rawDf.csv` ⇒ got raw genes (P=1610).
- CaDRReS MF pretrain → `prepare.py` loaded `Xtrain_kernel.csv` (line 443) but **passed the
  raw genes** to `Response_decompose` (line 449).
- KRR → kernel ✓ ; EN → raw genes ✓ (matches paper's per-drug EN on gene expression).

This explains the anomalies: EN looked strong (EN-on-genes = paper's EN), while PPO/CaDRReS
ran on a non-paper representation (raw genes, P=1610), inflating the reward scale (~85k) and
depressing their ranking.

### Fix (consumer-side, keeps EN on genes / KRR on kernel)

- `utils.read_FULL` + `read_PROP` (GDSC_ALL branch): read `Xtrain_kernel.csv` /
  `Xtest_kernel.csv` instead of `..._rawDf.csv`. (`--ess_genes_fn` is not passed in the PPO
  command, so no gene-name filter to break on kernel columns.)
- `prepare.py` MF pretrain: scale and pass `Xtrain_kernel`/`Xtest_kernel` to
  `Response_decompose` (was passing the raw-gene `Xtrain_df`).
- Net effect: PPO and CaDRReS now use the kernel (P = n_train ≈ 769, WP = 769×f), aligned;
  EN unchanged (genes), KRR unchanged (kernel). `prepare.py` CV split untouched — kernel
  files already exist, only MF pretrain + PPO need re-running.

### Re-run procedure (server)

Delete stale MF weights + PPO results to defeat the skip-if-done guards, then re-run:

```bash
rm -rf data/GDSC_ALL/CV/FULL/Fold*/100Dim            # MF weights → force re-pretrain on kernel
rm -f  results/data/GDSC_ALL/FULL/100Dim/CaDRRes/CaDRRes_*.npz
rm -f  results/data/GDSC_ALL/FULL/100Dim/ppo/ppo_*.npz
rm -rf Saved/ppo_data_GDSC_ALL_FULL_*                # old checkpoints
bash run_pipeline.sh                                 # STEP2 decompose-only + STEP3 PPO + STEP4
python results.py --config configs/configG_FULL_compare.yaml   # 4-method table
```

EN/KRR results are unaffected and need not be recomputed.

---

## PPORank hyperparameter/architecture audit vs §3.1.3 (2026-06-13)

Cross-checked the paper's PPORank spec (§3.1.3) against `arguments.py` defaults — and the
defaults ARE what ran, since `run_pipeline.sh`'s `main.py` call passes only `num_processes,
Data, analysis, algo, f, normalize_y, fold, cuda_id` (everything else = default). Result:
**the GDSC run was NOT paper-faithful on the PPO side either.**

### PPO hyperparameters

| Param (code) | Paper §3.1.3 | Ran (default) | Status |
|---|---|---|---|
| actors `num_processes` | 16 (GDSC) | 16 | ✓ |
| PPO epoch K `ppo_epoch` | 8 | 4 | ❌ |
| mini-batch size | 16 | ≈ T//`num_mini_batch` = 223//4 ≈ 55 | ❌ |
| clip ε `clip_param` | tune {0.1,0.2,0.3} | 0.2 | ~ok (in range) |
| λ `gae_lambda` | 0.95 (full-rank) | 0.95 | ✓ |
| γ `gamma` | 0.95 (full-rank) | 0.99 | ❌ |
| c1 `value_loss_coef` | 0.5 | 0.5 | ✓ |
| c2 `entropy_coef` | 0.001 | 0 | ❌ |

**Mini-batch — resolved as NOT a deviation.** Traced the rollout: the DataLoader batches
`num_processes`=16 cell lines (main.py:120); `rollouts.steps` *accumulates* every valid
ranking step across them (storage.py:97,181) so at update time `batch_size ≈ 16 × n_drugs ≈
3568`, and `feed_forward_generator` splits it into `num_mini_batch` chunks (≈892 transitions
at default 4). So `num_mini_batch` sub-splits the transition buffer, NOT cell lines, and does
NOT map to the paper's "mini-batch size 16" — which corresponds to the 16 cell lines per
update (= `num_processes`, already matched). Left `num_mini_batch` at default 4.

### Architecture (Deep & Cross actor)

| Param | Paper | Ran | Status |
|---|---|---|---|
| cross layers `nlayers_cross` | 2 | 1 | ❌ |
| deep net | 3 layers 128→64→32 | 256→128→64 (`deep_hidden_sizes=[256,128]`, `deep_out_size=64`) | ❌ |
| mini-batch (cell lines/update) | 16 | 16 (`num_processes`) | ✓ (see note above) |

`DeepNet(input,out,n_layers,hidden_sizes)` (DNN_models.py:90) builds `Linear(x0,h[0])→…→
Linear(h[-1],out)`. To match the paper: `deep_hidden_sizes=[128,64]`, `deep_out_size=32`
(keep `nlayers_deep=2`). ⚠️ `deep_out_size` is coupled to `f` via the critic constraint
`f ≥ deep_out_size + 4` (derived below); 32 ⇒ f≥36, fine at f=100, but smoke-tested anyway.

### Unclear / data-driven
- **`f` (MF/projection dim)**: paper states CaDRReS latent = 10 but does NOT give PPORank's
  `f` explicitly; we use 100. Possible deviation — flag, can't resolve from text.
- **T** (drugs/trajectory): paper says 265 (GDSC); ours = M = 223 (post toxic filter). Data-driven.
- **lr / seed**: not specified in paper (lr 3e-4, seed default).

### Fix APPLIED to `run_pipeline.sh` main.py call (GDSC), f kept at 100
`--ppo_epoch 8  --gamma 0.95  --entropy_coef 0.001  --nlayers_cross 2
--deep_hidden_sizes 128 64  --deep_out_size 32`
(num_processes 16 unchanged; `num_mini_batch` left at default 4 — see mini-batch note above.)

**Smoke-test PASSED (2026-06-13)**: ran `run_pipeline_quick.sh` (aligned to the same flags)
on the server. Confirmed end-to-end: kernel features active (`P=481` = n_train, not 1610),
`deep_out_size=32` + 2 cross layers build and the PPO update loop runs (epochs 0–3 completed,
no `ConvValueNet` reshape crash), total params 138638 (was 299132). test_ndcg climbed
0.664→0.706 in 3 epochs. Config validated → cleared for the full `run_pipeline.sh` (~8h).
(Quick uses 265 drugs — no `GDSC_drugMedianGE0.txt` in the QUICK dir; the full run filters to 223.)

---

## EN baseline switched to the Pearson kernel (2026-06-13)

Same root issue as the PPO feature bug, on the baseline side: `baselines.py` trained EN on
the **raw 1610-gene expression** while KRR (and now PPO/CaDRReS) use the **Pearson kernel**.
Per the paper's fair-comparison setup (§3.1.2 "same cell line features"; §3.1.1 defines those
as the Pearson correlation features; §3.1.2 names KRL as the lone method using raw genes), EN
should use the kernel too. EN-on-genes is almost certainly why EN was anomalously strong (0.778,
beating PPORank — contradicting the paper where EN≈CaDRReS≈PPORank).

Changes (`baselines.py` + `configG_FULL_compare.yaml`):
- EN now consumes `Xtrain_kernel.csv`/`Xtest_kernel.csv` (train kernel n_train×n_train; test
  kernel n_test×n_train); final fit on the full train kernel, predict on the test kernel.
- New `tune_en_kernel` restricts kernel columns to the inner-train set during validation
  (`Kvi = Ktr[val, inner]`), mirroring `tune_krr` — no leakage in hyperparameter selection.
- `l1_ratio` fixed to **0.5** (paper §3.1.2); only `alpha` tuned. KRR unchanged (already kernel).

Re-run command (kernel files already on disk, regenerated by the in-progress --decompose):
```bash
python baselines.py --config configs/configG_FULL_compare.yaml --methods EN --overwrite
```
Hypothesis to confirm: EN-on-kernel drops to ≈ PPORank/CaDRReS, resolving the anomaly.

---

## Why f ≥ deep_out_size + 4 (exact ConvValueNet derivation, 2026-06-13)

The projection dim `f` (CaDRReS latent dim; `WP` is `P×f`, drug embedding `M×f`) is *not*
free: the critic architecture imposes a lower bound. Derivation from `models/Policy.py`:

- Critic = `ConvValueNet(M, 1, kernel_size=cell_dim)` (Policy.py:200); `cell_dim = WP.shape[1] = f`.
  So it is `Conv1d(in=M, out=1, kernel=f)` then `AvgPool1d(kernel=f, stride=f)`, then sigmoid.
- Critic input = `in_final` (Deep_Cross_Policy.forward, line 101/120), last dim:
  `L = deep_out_size + x0_dim + 1`, with `x0_dim = drug_dim + cell_dim + 1 = 2f + 1`
  ⇒ **L = deep_out_size + 2f + 2**.
- Conv1d(kernel=f): output length `L − f + 1 = deep_out_size + f + 3`.
- AvgPool1d(kernel=f, stride=f): output length `floor((deep_out_size + 3)/f) + 1`.

The value head must emit **one scalar per state**, i.e. pooling output length = 1:
```
floor((deep_out_size + 3) / f) = 0   ⟺   f > deep_out_size + 3   ⟺   f ≥ deep_out_size + 4
```

Worked values (deep_out_size=32):
| f | pooling output length | critic |
|---|---|---|
| 100 | floor(35/100)+1 = 1 | ✓ scalar |
| 36  | floor(35/36)+1 = 1 | ✓ scalar (boundary) |
| 10  | floor(35/10)+1 = 4 | ✗ 4 values/state → shape clash with scalar returns in value loss |

So **f=10 is NOT runnable with the paper's deep net (deep_out_size=32 ⇒ needs f≥36)** —
it would need `deep_out_size ≤ 6` or a rewritten critic. This is why we keep **f=100** for
PPORank (and its CaDRReS warm-start). A paper-faithful CaDRReS at f=10 is only possible as a
*standalone* baseline (it doesn't use `ConvValueNet`), not as the PPO warm-start. Note: the
earlier "f≥68 / sample_concatenate reshape" note was imprecise — `sample_concatenate`'s
`obs_critic` block is commented out; the real bound is purely the `ConvValueNet` pooling above.

---

## GPU memory blow-up (~78GB) — diagnosed as allocator fragmentation (2026-06-13)

After the feature+hyperparameter fixes, the PPO run showed **~78 GB GPU reserved** (nvitop)
vs **<10 GB** on the old code — alarming on the shared single-GPU server. Static code reading
said the opposite should happen (kernel P=769 < genes 1610; smaller deep/cross nets), so the
78 GB couldn't be genuine need.

**Diagnosis (instrumented, not guessed):** added per-epoch `mem_alloc` (max_memory_allocated,
true need) vs `mem_resv` (max_memory_reserved, = what nvitop shows) to `main.py`. Re-ran with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Epoch 0 logged **`mem_alloc=1.6G
mem_resv=2.6G`** → the real need is ~1.6 GB; the 78 GB was the PyTorch caching allocator
**over-reserving due to fragmentation** (the new tensor-size pattern from kernel features +
new arch + drop_last mini-batches fragmented worse than the old config). Not a leak, not a bug.

**Fix:** `expandable_segments:True` makes reserved ≈ allocated (78 GB → 2.6 GB). Baked in as
the default `PYTORCH_CUDA_ALLOC_CONF` in `run_pipeline.sh` + `_quick.sh` (overridable).

**Also added (shared-GPU safety):** `--gpu_mem_fraction` (arguments.py/main.py) hard-caps this
process's GPU memory as a fraction of total; if exceeded, ONLY this process OOMs — never a
co-tenant's job. Wire via `GPU_MEM_FRACTION=0.5` env in `run_pipeline.sh`. Used as a safety
belt while a colleague was on the GPU (14 GB); with the allocator fix it's not even approached.

Note: epoch time rose (~90s → ~226s) while sharing the GPU with another job — compute
contention, not the fix; reverts when the GPU is free. Early stopping still ends folds early.

---

## Open issues / to verify

| Issue | Priority | Notes |
|-------|----------|-------|
| **PPO hyperparams ≠ paper §3.1.3** | ✅ fixed + smoke-tested | Run used defaults: K=4→8, γ=0.99→0.95, c2=0→0.001, cross layers 1→2, deep 256→128→64 ⇒ 128→64→32. Applied in `run_pipeline.sh` + `_quick.sh`. Smoke-test passed (deep_out_size=32 + kernel OK). Ready for full re-run |
| **EN anomalously strong** | ✅ fix applied (re-run pending) | Cause: EN ran on raw genes (1610), a different/more expressive space than the kernel used by the ranking methods. Paper §3.1.2 "fair comparison ... same cell line features" ⇒ EN should use the Pearson kernel like CaDRReS/PPORank/KRR (only KRL uses raw genes). Switched EN→kernel in `baselines.py` + fixed l1_ratio=0.5 (config). Re-run EN to confirm it drops to ≈PPORank/CaDRReS |
| **PPO saturates early** | High | PPORank 0.709 beats its CaDRReS warm-start (0.668, +0.041) but plateaus at ep ~15–20 and stays below EN. Check advantage/reward normalization, actor LR/entropy |
| **CaDRReS dim mismatch** | Medium | Paper sets CaDRReS latent dim=10; our MF pretrain uses f=100 (matches PPO). Likely why our CaDRReS is 3rd (0.668) not 2nd. Deviation to flag in report |
| **KRL baseline (Exp 1)** | High | Not in released code; must implement (RBF kernel over all 17737 GEX genes, λ×γ tuning) like EN/KRR in `baselines.py`. CaDRReS only needs aggregation (`.npz` already written by MF pretrain) |
| **SRMF baseline (Exp 1)** | Low | Paper lists it in Fig 3 but says it can't predict unseen cell lines; scoring method under 5-fold CV unclear — verify before attempting |
| **PPO-w/o (Exp 7/8 only)** | Medium | Reward ablation (drop positive eval signal), simulations only — NOT Exp 1, NOT the `--pretrain` flag. Implement when doing simulations |
| **MET modality** | Low | Cannot reproduce GEX+MET experiment (Figure 5 incomplete) — GDSC MET permanently lost |
| **`prepare.py` not yet run** | High | Must run before any training; no fold splits found in `data/GDSC_ALL/CV/` |
| **Toxic drug filtering** | ✅ resolved (GDSC) | GDSC filtered 265→223 via CaDRReS `GDSC_drugMedianGE0.txt` + new filter in `prepare.py` (see section above). CCLE (24→19) still to apply when CCLE config is created. Criterion differs from paper's Iorio+PubChem description — documented caveat |
| **GDSC vs GDSC_ALL** | Medium | `config.yaml` uses `data: GDSC`; `configG_FULL.yaml` uses `data: GDSC_ALL`; check if they refer to the same preprocessed data or to two different subsets |
| **CCLE config** | Medium | No `configC.yaml` exists; need to create it with correct dataset path, methods, and hyperparameters |
| **SimuData config** | Medium | No config for simulation experiments; need to create it |
| **CaDRRes preprocessing** | High | `preprocess_fts_cl_drug.py` must be run before training to generate CaDRRes input features; not yet verified |
| **SimuData: `prepare_simu.py` requires PyTorch** | Medium | `prepare_simu.py` imports `utils.py` which imports torch; use `split_simu_data.py` (standalone) for CV splits; `prepare_simu.py` is only needed for CaDRRes pretraining on SimuData |
| **TCGA BRCA data** | ✅ resolved | `TCGA_BRCA.npz` and `TCGA_BRCA_clinical.csv.gz` generated 2026-06-09 |
| **SimuData** | ✅ resolved | N=1000 and N=10000 generated 2026-06-09; CV splits in `data/SimuData/` |
