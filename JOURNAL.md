# Reproduction Journal — PPORank

**Paper**: PPORank (see `paper.pdf`)  
**Original repo**: https://github.com/mylzwq/PPORank  
**Goal**: Reproduce the paper results on the GDSC dataset.

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

## Execution pipeline (GDSC FULL)

```bash
# 1. Activate environment
conda activate pporank

# 2. Generate .npz files from raw data
python preprocess/load_dataset.py

# 3. Generate cross-validation fold splits
python prepare.py --config configs/configG_FULL.yaml

# 4. Training
python main.py --config configs/configG_FULL.yaml
```

---

## Open issues / to verify

- The MET file (`METH_CELLLINES_BEMs/PANCAN.txt`) is unavailable: the pipeline runs without it (GEX, WES, CNV only). The main config `configG_FULL.yaml` uses `Data_All: True` and `data: GDSC_ALL`, which in practice loads only GEX — so the missing MET does not impact reproduction.
- The config includes CaDRRes as a baseline method (`methods: ['KRR','KRL','CaDRRes','EN']`): verify that `preprocess/preprocess_fts_cl_drug.py` runs correctly before training.
- `prepare.py` has not been run yet: this is the next required step before training.
