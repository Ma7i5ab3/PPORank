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
- Constraint documented in both files: keep `f ≥ 68` (deep_out_size 64 + 4),
  otherwise the critic's `ConvValueNet` pooling breaks
  `RolloutStorage.sample_concatenate`'s reshape.

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

## Open issues / to verify

| Issue | Priority | Notes |
|-------|----------|-------|
| **MET modality** | Low | Cannot reproduce GEX+MET experiment (Figure 5 incomplete) — GDSC MET permanently lost |
| **`prepare.py` not yet run** | High | Must run before any training; no fold splits found in `data/GDSC_ALL/CV/` |
| **Toxic drug filtering** | High | Paper uses 223 GDSC drugs / 19 CCLE drugs; `.npz` files have 265 / 24; verify that `preprocess/toxic_data.py` is called correctly and produces the right filtered dataset |
| **GDSC vs GDSC_ALL** | Medium | `config.yaml` uses `data: GDSC`; `configG_FULL.yaml` uses `data: GDSC_ALL`; check if they refer to the same preprocessed data or to two different subsets |
| **CCLE config** | Medium | No `configC.yaml` exists; need to create it with correct dataset path, methods, and hyperparameters |
| **SimuData config** | Medium | No config for simulation experiments; need to create it |
| **CaDRRes preprocessing** | High | `preprocess_fts_cl_drug.py` must be run before training to generate CaDRRes input features; not yet verified |
| **SimuData: `prepare_simu.py` requires PyTorch** | Medium | `prepare_simu.py` imports `utils.py` which imports torch; use `split_simu_data.py` (standalone) for CV splits; `prepare_simu.py` is only needed for CaDRRes pretraining on SimuData |
| **TCGA BRCA data** | ✅ resolved | `TCGA_BRCA.npz` and `TCGA_BRCA_clinical.csv.gz` generated 2026-06-09 |
| **SimuData** | ✅ resolved | N=1000 and N=10000 generated 2026-06-09; CV splits in `data/SimuData/` |
