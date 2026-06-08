"""
create_TCGA_clinical.py
Assembles data/GDSC_ALL/TCGA_BRCA_clinical.csv.gz from four Firehose/external sources.

Required columns (as used by results_TCGA.py):
  ER            — string: 'positive' / 'negative' / 'indeterminate' / 'equivocal'
  PR            — string: same values
  HER2          — string: same values
  CHR17         — float:  weighted-mean Segment_Mean for chr17 (log2 ratio; NaN if absent)
  BRCA_germline — bool:   True for germline BRCA1/2 loss-of-function carriers
  JAK2_RPPA     — float:  JAK2 RPPA expression level (NaN if absent)

Index: patient ID, 12-char TCGA format (e.g. TCGA-3C-AAAU)

─────────────────────────────────────────────────────────────────────────────
DOWNLOAD STEPS (run once from data/GDSC_ALL/):
─────────────────────────────────────────────────────────────────────────────
BASE="http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/BRCA/20160128"

# 1. Clinical (ER/PR/HER2) — ~0.5 MB
curl -O "$BASE/gdac.broadinstitute.org_BRCA.Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz"
tar -xzf gdac.broadinstitute.org_BRCA.Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz

# 2. RPPA (JAK2) — ~1.9 MB
curl -O "$BASE/gdac.broadinstitute.org_BRCA.RPPA_AnnotateWithGene.Level_3.2016012800.0.0.tar.gz"
tar -xzf gdac.broadinstitute.org_BRCA.RPPA_AnnotateWithGene.Level_3.2016012800.0.0.tar.gz

# 3. CNV segments hg19 (CHR17) — ~18 MB
curl -O "$BASE/gdac.broadinstitute.org_BRCA.Merge_snp__genome_wide_snp_6__broad_mit_edu__Level_3__segmented_scna_hg19__seg.Level_3.2016012800.0.0.tar.gz"
tar -xzf "gdac.broadinstitute.org_BRCA.Merge_snp__genome_wide_snp_6__broad_mit_edu__Level_3__segmented_scna_hg19__seg.Level_3.2016012800.0.0.tar.gz"

# 4. Maxwell et al. Nat Commun 2017 — MANUAL DOWNLOAD
#    Go to: https://www.nature.com/articles/s41467-017-00388-9#Sec22
#    Download "Supplementary Data 1" and save as:
#      data/GDSC_ALL/maxwell2017_supp_data1.xlsx
#    (or .csv — adjust MAXWELL_FILE below)
─────────────────────────────────────────────────────────────────────────────

Run from PPORank project root:
  conda activate pporank
  python preprocess/create_TCGA_clinical.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'GDSC_ALL'
)

# ─── output ───────────────────────────────────────────────────────────────────
OUT_FILE = os.path.join(DATA_DIR, 'TCGA_BRCA_clinical.csv.gz')

# ─── Maxwell et al. supplementary data 1 ─────────────────────────────────────
# Adjust filename if downloaded as .csv
MAXWELL_FILE = os.path.join(DATA_DIR, 'maxwell2017_supp_data1.xlsx')


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def find_file(directory, *patterns, recursive=True):
    """Return the first file in `directory` matching any glob pattern."""
    for pat in patterns:
        hits = glob.glob(os.path.join(directory, '**', pat) if recursive
                         else os.path.join(directory, pat),
                         recursive=recursive)
        if hits:
            return hits[0]
    return None


def patient_id(barcode):
    """Normalise a full TCGA barcode to the 12-char patient ID."""
    return str(barcode).upper().replace('.', '-')[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — ER / PR / HER2 from Firehose Clinical_Pick_Tier1
# ─────────────────────────────────────────────────────────────────────────────
def load_clinical_ihc(data_dir):
    """
    Returns a DataFrame indexed by patient ID (TCGA-XX-XXXX) with columns
    ER, PR, HER2 (lowercase string values matching results_TCGA.py constants).

    Firehose Clinical_Pick_Tier1 files have patients as COLUMNS and variables
    as rows; patient IDs are lowercase (e.g. tcga-3c-aaau).
    """
    clin_dir = find_file(
        data_dir,
        'gdac.broadinstitute.org_BRCA.Clinical_Pick_Tier1.Level_4.*',
        recursive=False
    )
    if clin_dir is None:
        # Try searching for the extracted directory
        clin_dir = os.path.join(
            data_dir,
            'gdac.broadinstitute.org_BRCA.Clinical_Pick_Tier1.Level_4.2016012800.0.0'
        )

    clin_file = find_file(clin_dir or data_dir,
                          'BRCA.clin.merged.picked.txt',
                          'BRCA.merged.picked.txt',
                          '*.clin.merged.picked.txt')
    if clin_file is None:
        raise FileNotFoundError(
            "Clinical_Pick_Tier1 file not found. "
            "Download and extract the Firehose Clinical_Pick_Tier1 archive into:\n"
            f"  {data_dir}"
        )

    print(f"[IHC] Loading clinical data from: {os.path.basename(clin_file)}")
    # Firehose format: rows = variable names, cols = patient IDs (lowercase)
    clin = pd.read_csv(clin_file, sep='\t', index_col=0, low_memory=False)
    # Transpose so patients are rows
    clin = clin.T
    clin.index = clin.index.str.upper().str.replace('.', '-', regex=False)

    # Locate ER / PR / HER2 columns (case-insensitive)
    col_map = {c.lower(): c for c in clin.columns}

    def get_col(*candidates):
        for c in candidates:
            if c in col_map:
                return col_map[c]
        return None

    er_col  = get_col('er_status_by_ihc',  'breast_carcinoma_estrogen_receptor_status')
    pr_col  = get_col('pr_status_by_ihc',  'breast_carcinoma_progesterone_receptor_status')
    her_col = get_col('ihc_her2',          'lab_proc_her2_neu_immunohistochemistry_receptor_status',
                      'her2_status_by_ihc')

    missing = [n for n, c in [('ER', er_col), ('PR', pr_col), ('HER2', her_col)] if c is None]
    if missing:
        print(f"  Available columns (sample): {list(clin.columns[:20])}")
        raise KeyError(f"Could not find IHC columns: {missing}")

    df = pd.DataFrame({
        'ER':   clin[er_col],
        'PR':   clin[pr_col],
        'HER2': clin[her_col],
    })
    # Normalise values to lowercase (results_TCGA.py uses 'positive'/'negative')
    for col in ['ER', 'PR', 'HER2']:
        df[col] = df[col].str.lower().str.strip()

    print(f"  IHC data: {len(df)} patients")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — CHR17 from Firehose SNP6 segment copy-number (hg19)
# ─────────────────────────────────────────────────────────────────────────────
def load_chr17(data_dir):
    """
    Returns a Series indexed by patient ID with the weighted-mean Segment_Mean
    (log2 copy-number ratio) over all chromosome-17 segments.

    results_TCGA.py uses: chr17_pos = (CHR17 >= 2)
    i.e. chr17_pos is True when log2(CN/2) >= 2 → absolute CN >= 8.
    This flags patients with high-level chr17 amplification (HER2 amplicon).
    """
    seg_dir_pattern = (
        'gdac.broadinstitute.org_BRCA.Merge_snp__genome_wide_snp_6__'
        'broad_mit_edu__Level_3__segmented_scna_hg19__seg.Level_3.*'
    )
    seg_dir = find_file(data_dir, seg_dir_pattern, recursive=False)
    seg_file = find_file(seg_dir or data_dir,
                         'BRCA.snp__genome_wide_snp_6__broad_mit_edu__Level_3__segmented_scna_hg19__seg.seg.txt',
                         '*.hg19.seg.txt',
                         '*.seg.txt')
    if seg_file is None:
        raise FileNotFoundError(
            "SNP6 hg19 segment file not found. "
            "Download and extract the Firehose segmented_scna_hg19 archive into:\n"
            f"  {data_dir}"
        )

    print(f"[CHR17] Loading CNV segments from: {os.path.basename(seg_file)}")
    seg = pd.read_csv(seg_file, sep='\t', low_memory=False)

    # Normalise column names (Firehose uses mixed capitalisation)
    seg.columns = seg.columns.str.strip().str.lower()
    # Expected columns: sample, chromosome, start, end, num_probes (or num.markers), segment_mean
    seg = seg.rename(columns={
        'chrom':        'chromosome',
        'loc.start':    'start',
        'loc.end':      'end',
        'num.mark':     'num_probes',
        'num_mark':     'num_probes',
        'seg.mean':     'segment_mean',
        'segmented_mean': 'segment_mean',
    })

    # Filter chromosome 17
    chr17 = seg[seg['chromosome'].astype(str).isin(['17', 'chr17'])].copy()

    # Patient ID: first 12 chars of barcode
    chr17['patient'] = chr17['sample'].apply(patient_id)

    # Weighted mean Segment_Mean per patient
    def weighted_mean(group):
        w = group['num_probes'].astype(float)
        v = group['segment_mean'].astype(float)
        valid = w.notna() & v.notna()
        if valid.sum() == 0:
            return np.nan
        return np.average(v[valid], weights=w[valid])

    chr17_series = chr17.groupby('patient').apply(weighted_mean)
    chr17_series.name = 'CHR17'

    n_pos = (chr17_series >= 2).sum()
    print(f"  CHR17 data: {len(chr17_series)} patients  "
          f"(chr17_pos [>=2]: {n_pos})")
    return chr17_series


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — BRCA_germline from Maxwell et al. Nat Commun 2017
# ─────────────────────────────────────────────────────────────────────────────
def load_brca_germline(maxwell_file):
    """
    Returns a set of 12-char patient IDs that carry germline BRCA1/2
    loss-of-function mutations (Maxwell et al. 2017, Supplementary Data 1).

    The supplementary file lists TCGA sample barcodes (one per row or in a
    column); this function searches all string columns for TCGA-style IDs.
    """
    if not os.path.exists(maxwell_file):
        print(f"[BRCA_germline] WARNING: {maxwell_file} not found.")
        print("  Download Supplementary Data 1 from:")
        print("  https://www.nature.com/articles/s41467-017-00388-9#Sec22")
        print("  and save it as:", maxwell_file)
        print("  BRCA_germline column will be set to False for all patients.")
        return set()

    print(f"[BRCA_germline] Loading: {os.path.basename(maxwell_file)}")
    ext = os.path.splitext(maxwell_file)[1].lower()
    if ext in ('.xlsx', '.xls'):
        raw = pd.read_excel(maxwell_file, header=0)
    else:
        raw = pd.read_csv(maxwell_file, header=0, sep=None, engine='python')

    # Search all string columns for TCGA barcodes (pattern: TCGA-XX-XXXX)
    carriers = set()
    for col in raw.columns:
        vals = raw[col].dropna().astype(str)
        tcga_ids = vals[vals.str.match(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', na=False)]
        for v in tcga_ids:
            carriers.add(patient_id(v))

    # Keep only BRCA (breast) samples: patient IDs from TCGA BRCA cohort
    # (ovarian carriers are also in Maxwell; they will be filtered later by
    #  intersection with test_ids from TCGA_BRCA.npz, but we flag all for now)
    print(f"  Found {len(carriers)} germline BRCA1/2 carriers (TCGA IDs)")
    return carriers


# ─────────────────────────────────────────────────────────────────────────────
# Source 4 — JAK2_RPPA from Firehose RPPA_AnnotateWithGene Level 3
# ─────────────────────────────────────────────────────────────────────────────
def load_jak2_rppa(data_dir):
    """
    Returns a Series indexed by patient ID with JAK2 RPPA expression values.
    results_TCGA.py uses: jak2_pos = (~np.isnan(x) and x >= 0)
    """
    rppa_dir = find_file(
        data_dir,
        'gdac.broadinstitute.org_BRCA.RPPA_AnnotateWithGene.Level_3.*',
        recursive=False
    )
    rppa_file = find_file(rppa_dir or data_dir,
                          'BRCA.rppa.txt',
                          'BRCA_RPPA_data.txt',
                          '*.rppa.txt')
    if rppa_file is None:
        raise FileNotFoundError(
            "RPPA file not found. "
            "Download and extract the Firehose RPPA_AnnotateWithGene archive into:\n"
            f"  {data_dir}"
        )

    print(f"[JAK2_RPPA] Loading RPPA from: {os.path.basename(rppa_file)}")
    # Firehose RPPA: rows = proteins (format "GENE|antibody"), cols = sample barcodes
    rppa = pd.read_csv(rppa_file, sep='\t', index_col=0)

    # Find JAK2 row
    jak2_rows = [i for i in rppa.index if str(i).upper().startswith('JAK2')]
    if not jak2_rows:
        print(f"  WARNING: JAK2 not found in RPPA. Available proteins (sample): "
              f"{list(rppa.index[:10])}")
        return pd.Series(dtype=float, name='JAK2_RPPA')

    jak2_vals = rppa.loc[jak2_rows[0]]
    jak2_vals.index = jak2_vals.index.map(patient_id)
    jak2_vals.name = 'JAK2_RPPA'

    n_pos = (jak2_vals >= 0).sum()
    print(f"  JAK2_RPPA: {len(jak2_vals)} samples  (jak2_pos [>=0]: {n_pos})")
    return jak2_vals


# ─────────────────────────────────────────────────────────────────────────────
# Assemble
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # --- Load each source ---
    ihc_df        = load_clinical_ihc(DATA_DIR)
    chr17_series  = load_chr17(DATA_DIR)
    brca_carriers = load_brca_germline(MAXWELL_FILE)
    jak2_series   = load_jak2_rppa(DATA_DIR)

    # --- Union of all patient IDs ---
    all_ids = (set(ihc_df.index)
               | set(chr17_series.index)
               | set(jak2_series.index))
    print(f"\nTotal unique patient IDs across all sources: {len(all_ids)}")

    # --- Build output DataFrame ---
    out = pd.DataFrame(index=sorted(all_ids))
    out.index.name = 'patient_id'

    # IHC columns
    out = out.join(ihc_df[['ER', 'PR', 'HER2']], how='left')

    # CHR17
    out = out.join(chr17_series.rename('CHR17'), how='left')

    # BRCA_germline
    out['BRCA_germline'] = out.index.isin(brca_carriers)

    # JAK2_RPPA
    out = out.join(jak2_series.rename('JAK2_RPPA'), how='left')

    # --- Summary ---
    print("\nColumn fill rates:")
    for col in ['ER', 'PR', 'HER2', 'CHR17', 'BRCA_germline', 'JAK2_RPPA']:
        n_valid = out[col].notna().sum() if out[col].dtype != bool else out[col].sum()
        print(f"  {col:15s}: {n_valid:4d} / {len(out)}")

    # Quick sanity check against paper results (Section 3.4)
    her2_pos   = (out['HER2'] == 'positive').sum()
    brca_count = out['BRCA_germline'].sum()
    print(f"\nSanity check (paper values in parentheses):")
    print(f"  HER2+:         {her2_pos:4d}  (paper: 163)")
    print(f"  BRCA_germline: {brca_count:4d}  (paper: 37)")

    # --- Save ---
    out.to_csv(OUT_FILE, compression='gzip')
    print(f"\nSaved: {OUT_FILE}  ({len(out)} patients x {len(out.columns)} columns)")


if __name__ == '__main__':
    main()
