"""
create_TCGA_data.py  —  Python 3 port of BorgwardtLab/Kernelized-Rank-Learning/create_TCGA_data.py

Assembles TCGA_BRCA.npz from:
  - pRRophetic_TCGA_BRCA_{trainFrame,testFrame,preds}.csv.gz  (from breast_cancer_analysis.R)
  - data/GDSC_ALL/GDSC_GEX.npz                               (from load_dataset.py)

Output:
  data/GDSC_ALL/TCGA_BRCA.npz
    Keys: train_X, train_Y, drug_ids, drug_names, test_X, test_ids

Run from PPORank project root:
  python preprocess/create_TCGA_data.py
"""

import numpy as np
import os
import sys

# Ensure project root is on the path so misc.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from misc import LAPATINIB, RUXOLITINIB
from misc import DRUG_IDS, TNBC_DRUGS
from misc import load_pRRophetic_data, intersect_index

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'GDSC_ALL')
np.random.seed(0)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load GDSC IC50 matrix
# ─────────────────────────────────────────────────────────────────────────────
gdsc_gex      = np.load(os.path.join(DATA_DIR, 'GDSC_GEX.npz'))
gdsc_X        = gdsc_gex['X']
gdsc_Y        = gdsc_gex['Y']
gdsc_drug_ids = gdsc_gex['drug_ids'].astype(int)
gdsc_drug_names = gdsc_gex['drug_names']
gdsc_samples  = gdsc_gex['cell_names']
print(f'gdsc_X: {gdsc_X.shape}  gdsc_Y: {gdsc_Y.shape}')

# Column index of each BRCA drug of interest in the GDSC IC50 matrix
drug_index = {}
for drug in [LAPATINIB] + TNBC_DRUGS:
    hits = np.nonzero(gdsc_drug_ids == int(DRUG_IDS[drug]))[0]
    if len(hits) == 0:
        raise ValueError(f"Drug '{drug}' (ID={DRUG_IDS[drug]}) not found in GDSC_GEX.npz")
    drug_index[drug] = int(hits[0])

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load pRRophetic-harmonized expression
#   train_samples = GDSC cell lines (from pRRophetic's internal GDSC data)
#   test_samples  = TCGA patient IDs
# ─────────────────────────────────────────────────────────────────────────────
train_X, _, train_samples, test_X, _, test_samples = load_pRRophetic_data(
    data_dir=DATA_DIR,
    fn_prefix='pRRophetic_TCGA_BRCA',
    test_y_suffix=None
)
print(f'pRRophetic train: {train_X.shape}  samples: {train_samples.shape}')
print(f'pRRophetic test:  {test_X.shape}   samples: {test_samples.shape}')

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Align pRRophetic cell lines with GDSC_GEX cell lines
#   pRRophetic uses its own GDSC copy; cell line names may differ in formatting.
#   Normalise both sides: lowercase, strip leading 'x' (R prefix), strip whitespace.
# ─────────────────────────────────────────────────────────────────────────────
train_samples_norm = np.array([x.lower().strip('x').strip() for x in train_samples])
gdsc_samples_norm  = np.array([x.lower().strip()             for x in gdsc_samples])

merged = intersect_index(train_samples_norm, gdsc_samples_norm)
X_keep = np.array(merged['index1'].values, dtype=int)
Y_keep = np.array(merged['index2'].values, dtype=int)

train_X = train_X[X_keep]
train_Y = gdsc_Y[Y_keep, :]
print(f'After cell-line intersection: {train_X.shape[0]} cell lines')

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Filter TCGA: keep only primary solid tumors (01), first aliquot (A)
#   TCGA barcode format: TCGA-XX-XXXX-01A-...
#   Positions 13–15 encode the sample type (01A = primary solid tumor, first aliquot)
#   Positions 0–11 are the patient identifier (12 chars)
# ─────────────────────────────────────────────────────────────────────────────
sample_types = np.array([x[13:15] for x in test_samples])
test_samples = np.array([x.upper().replace('.', '-')[:12] for x in test_samples])
is_tumor     = sample_types == '01'
test_X       = test_X[is_tumor]
test_samples = test_samples[is_tumor]
print(f'After primary-tumor filter: {test_X.shape[0]} TCGA patients')

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Remove genes with NaN values in TCGA
# ─────────────────────────────────────────────────────────────────────────────
not_nan = ~np.any(np.isnan(test_X), axis=0)
test_X  = test_X[:, not_nan]
train_X = train_X[:, not_nan]

assert np.all(~np.isnan(test_X)),  "NaNs remain in test_X after filtering"
assert np.all(~np.isnan(train_X)), "NaNs remain in train_X after filtering"
assert np.all([not np.all(np.isnan(y)) for y in train_Y])
print(f'Genes after NaN filter: {test_X.shape[1]}')

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Reorder drug columns: BRCA drugs of interest first
#   (backwards compatibility with KRL results loading code)
# ─────────────────────────────────────────────────────────────────────────────
idx = list(range(train_Y.shape[1]))
for i, drug in enumerate([LAPATINIB] + TNBC_DRUGS):
    idx.remove(drug_index[drug])
    idx.insert(i, drug_index[drug])
    drug_index[drug] = i
train_Y = train_Y[:, idx]

# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Drop cell lines with ≤1 drug response recorded
#   (cannot contribute to learning-to-rank)
# ─────────────────────────────────────────────────────────────────────────────
enough_data = [np.sum(~np.isnan(y)) > 1 for y in train_Y]
train_Y = train_Y[enough_data]
train_X = train_X[enough_data]

print(f'GDSC CCLs (training data): X {train_X.shape}  Y {train_Y.shape}')
print(f'TCGA patients (test data): X {test_X.shape}   ids {test_samples.shape}')

# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Save
# ─────────────────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, 'TCGA_BRCA.npz')
np.savez(out_path,
         train_X=train_X,
         train_Y=train_Y,
         drug_ids=gdsc_drug_ids,
         drug_names=gdsc_drug_names,
         test_X=test_X,
         test_ids=test_samples)
print(f'Saved: {out_path}')
