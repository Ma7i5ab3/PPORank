#!/usr/bin/env python
"""
baselines_krl.py — Kernelized Rank Learning (KRL) baseline for the FULL
experiment (Claim 1).

Uses the *actual* KRL algorithm from He, Folkman & Borgwardt (Bioinformatics
2018): listwise NDCG loss optimised via the Bundle Method for Regularised Risk
Minimisation (BMRM) over an RBF kernel on all 17 737 gene-expression features.

INPUTS
------
GDSC_GEX.npz             raw 17737-gene expression + 265-drug sensitivity
CV/FULL/Fold{i}/
  YtrainDf.csv           train drug-sensitivity matrix  (cell_ids × drugs)
  YtestDf.csv            test  drug-sensitivity matrix  (cell_ids × drugs)

OUTPUTS
-------
For each fold i and each (lambda, gamma) combination:
    results/<data>/FULL/<f>Dim/KRL/KRL_{i}_lambda{l}_gamma{g}.npz
        keys: Y_true, Y_pred   (n_test_cells × n_drugs)

Lambda and gamma are tuned post-hoc by results.py at aggregation time.
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "KRL"))
from krl_core import KRL_fit, KRL_pred

from results import get_result_filename

logger = logging.getLogger("baselines_krl")

# Default grids (widened from the PPORank config to cover the original KRL paper)
KRL_LAMBDAS = [0.001, 0.01, 0.1, 1.0, 10.0]
KRL_GAMMAS = [0.001, 0.01, 0.1, 1.0, 10.0]


def setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, "baselines_krl_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logging.captureWarnings(True)
    warn_logger = logging.getLogger("py.warnings")
    warn_logger.addHandler(fh)
    logger.info("Logging to %s", log_path)
    return log_path


def parse_args():
    p = argparse.ArgumentParser(description="KRL baseline for the FULL experiment")
    p.add_argument("--config", type=str, required=True,
                   help="aggregation config (reads data, nfolds, krl_lambdas, krl_gammas)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even if the output .npz already exists")
    return p.parse_args()


def load_full_gex_gdsc(data_dir):
    npz_path = os.path.join(data_dir, "GDSC_GEX.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            "GDSC_GEX.npz not found at {}. Run load_dataset.py first.".format(npz_path))
    data = np.load(npz_path)
    return data['X'].astype(np.float64), data['cell_ids']


def load_full_gex_ccle(data_dir):
    expr_path = os.path.join(data_dir, "CCLE_expression.csv")
    if not os.path.exists(expr_path):
        raise FileNotFoundError(
            "CCLE_expression.csv not found at {}. Download from DepMap.".format(expr_path))
    meta_path = os.path.join(data_dir, "sample_info.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            "sample_info.csv not found at {}. Download from DepMap.".format(meta_path))

    expr = pd.read_csv(expr_path, index_col=0)          # (1376, 19177)  DepMap_ID x genes
    meta = pd.read_csv(meta_path)
    meta = meta[['DepMap_ID', 'CCLE_Name']].dropna(subset=['CCLE_Name'])
    # CCLE_Name column has format like "LN18_CENTRAL_NERVOUS_SYSTEM"
    meta['CCLE_short'] = meta['CCLE_Name'].str.split('_').str[0].str.lower()

    # Map: DepMap_ID -> CCLE_short, average any duplicates
    id_to_short = dict(zip(meta['DepMap_ID'], meta['CCLE_short']))
    expr.index = expr.index.map(id_to_short)
    expr = expr.dropna(how='all')                      # drop unmapped rows
    # Average rows with the same short name (rare duplicate DepMap -> same CCLE line)
    expr = expr.groupby(expr.index).mean()
    return expr.values.astype(np.float64), expr.index.values


def load_fold_y(fold_dir):
    ytr = pd.read_csv(os.path.join(fold_dir, "YtrainDf.csv"), index_col=0)
    yte = pd.read_csv(os.path.join(fold_dir, "YtestDf.csv"), index_col=0)
    return ytr, yte


def map_cell_ids_to_gex(full_gex, full_cell_ids, fold_cell_ids):
    # For GDSC: cell_ids are numeric strings (e.g. "1240121"), matching as-is.
    # For CCLE: fold uses full names (e.g. "LN18_CENTRAL_NERVOUS_SYSTEM"), expression
    # uses stripped-lower (e.g. "ln18").  Extract the first underscore-delimited part.
    full_index = pd.Index(full_cell_ids)
    fold_short = pd.Index(fold_cell_ids.astype(str)).str.split('_').str[0].str.lower()
    matched = full_index.get_indexer(fold_short)
    valid = matched >= 0
    if not valid.all():
        logger.warning("%d cell line(s) from fold not found in expression data", (~valid).sum())
    return full_gex[matched[valid]], valid


def main():
    args = parse_args()
    setup_logger()
    with open(args.config) as f:
        config = yaml.full_load(f)
    logger.info("config=%s | seed=%s", args.config, args.seed)

    analysis = config["analysis"]
    assert analysis == "FULL", "baselines_krl.py only handles the FULL analysis"
    data_name = config["data"]
    nfolds = config["nfolds"]
    f = config["f"]                     # used only for result path, not model dim

    # The config grids are often minimal placeholders; widen to the defaults
    # defined above if they look too narrow.
    lambdas = config.get("krl_lambdas") or KRL_LAMBDAS
    gammas = config.get("krl_gammas") or KRL_GAMMAS
    krl_k = config.get("krl_k", 10)     # NDCG@k truncation during training

    data_dir = os.path.join(os.getcwd(), data_name) if not data_name.startswith(os.sep) else data_name
    data_kind = os.path.basename(data_name)

    if data_kind.startswith("GDSC"):
        logger.info("Loading GDSC_GEX.npz from %s ...", data_dir)
        full_gex, full_cell_ids = load_full_gex_gdsc(data_dir)
    elif data_kind == "CCLE":
        logger.info("Loading CCLE_expression.csv from %s ...", data_dir)
        full_gex, full_cell_ids = load_full_gex_ccle(data_dir)
    else:
        raise ValueError("Unsupported dataset: {}".format(data_kind))
    logger.info("Full GEX: %d cell lines x %d genes", full_gex.shape[0], full_gex.shape[1])

    fold_root = os.path.join(data_dir, "CV", "FULL")

    total_start = time.time()
    for i in range(nfolds):
        fold_dir = os.path.join(fold_root, "Fold{}".format(i))
        fold_start = time.time()
        logger.info("=== Fold %d/%d ===", i, nfolds - 1)

        ytr_df, yte_df = load_fold_y(fold_dir)
        ytr = ytr_df.values.astype(np.float64)
        yte = yte_df.values.astype(np.float64)
        logger.info("  Ytrain: %s, Ytest: %s", ytr.shape, yte.shape)

        # Map fold cell IDs to raw GEX rows and standardise (z-score per gene)
        x_tr_raw, tr_keep = map_cell_ids_to_gex(full_gex, full_cell_ids, ytr_df.index.values)
        x_te_raw, te_keep = map_cell_ids_to_gex(full_gex, full_cell_ids, yte_df.index.values)
        ytr = ytr[tr_keep]
        yte = yte[te_keep]

        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_tr_raw)
        x_te = scaler.transform(x_te_raw)
        logger.info("  Xtrain: %s, Xtest: %s (standardised expression)", x_tr.shape, x_te.shape)

        for lam in lambdas:
            for gam in gammas:
                out = get_result_filename("KRL", "FULL", data_name, i, f,
                                          params=[lam, gam])
                if os.path.exists(out) and not args.overwrite:
                    logger.info("    lam=%s gam=%s: exists, skipping", lam, gam)
                    continue

                t1 = time.time()
                logger.info("    Training KRL (lam=%s, gam=%s, k=%d) ...", lam, gam, krl_k)
                W = KRL_fit(x_tr, ytr, k=krl_k, Lambda=lam, gamma=gam,
                            njobs=os.cpu_count() or 1, verbose=True)
                ypred = KRL_pred(W, x_tr, x_te, gamma=gam)
                np.savez(out, Y_true=yte, Y_pred=ypred)
                logger.info("    lam=%-6s gam=%-6s | %.1fs -> %s",
                            lam, gam, time.time() - t1, os.path.relpath(out))

        logger.info("  Fold %d done in %.1fs", i, time.time() - fold_start)

    logger.info("All %d folds complete in %.1fs", nfolds, time.time() - total_start)


if __name__ == "__main__":
    main()
