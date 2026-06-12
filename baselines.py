#!/usr/bin/env python
"""
baselines.py — ElasticNet (EN) and Kernel Ridge (KRR) baselines for the FULL
experiment (Claim 1), reusing the cross-validation folds produced by prepare.py.

WHY THIS FILE EXISTS
--------------------
The released PPORank code does NOT include training code for the EN/KRR/KRL
baselines: they appear only as name constants (Constant.py / misc.py) and inside
the result-aggregation logic (results.py). CaDRRes is produced as a by-product of
the matrix-factorization pretraining (prepare.py --decompose). This script
reimplements the two regression baselines (EN, KRR) with scikit-learn so the
FULL/Claim-1 comparison table can be completed.

IMPORTANT (for the reproducibility report): EN and KRR here are a faithful
reimplementation, NOT the authors' original baseline code (which was not
released). They are trained on exactly the same per-fold splits, features and
target matrices used by PPORank (and CaDRRes), so the comparison is fair.

INPUTS (written by prepare.py Split_Data, one folder per fold)
    <data>/CV/FULL/Fold{i}/
        Xtrain_rawDf.csv   raw cell-line features (gene expression)   [EN]
        Xtest_rawDf.csv
        Xtrain_kernel.csv  train x train Pearson kernel               [KRR]
        Xtest_kernel.csv   test  x train Pearson kernel               [KRR]
        YtrainDf.csv       drug-sensitivity matrix (-log IC50)
        YtestDf.csv

OUTPUTS (exact layout that results.py expects)
    results/<data>/FULL/<f>Dim/<method>/<method>_{fold}.npz
        keys: Y_true, Y_pred   (n_test_cells x n_drugs)

Hyperparameters are selected per fold by an internal validation split scored with
full-rank NDCG (the paper's evaluation metric), over the grids below.
"""
import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from results import get_result_filename
from Reward_utils import NDCGk

logger = logging.getLogger("baselines")


def setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, "baselines_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    # route warnings (e.g. sklearn ConvergenceWarning) into the same log
    logging.captureWarnings(True)
    warn_logger = logging.getLogger("py.warnings")
    warn_logger.addHandler(fh)
    logger.info("Logging to %s", log_path)
    return log_path

# Default tuning grids (override via --config grids or CLI). Kept small but
# sensible; the config's en_alphas/en_l1ratios are used if present.
EN_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]
EN_L1RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
KRR_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def parse_args():
    p = argparse.ArgumentParser(description="EN / KRR baselines for the FULL experiment")
    p.add_argument("--config", type=str, required=True,
                   help="aggregation/training config (reads data, nfolds, f, grids)")
    p.add_argument("--methods", nargs="+", default=["EN", "KRR"],
                   choices=["EN", "KRR"], help="which baselines to run")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="fraction of TRAIN cell lines held out for hyperparameter selection")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--min_samples", type=int, default=5,
                   help="skip a drug if it has fewer than this many non-NaN train labels")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even if the output .npz already exists")
    return p.parse_args()


def load_fold(fold_dir):
    def rd(name):
        return pd.read_csv(os.path.join(fold_dir, name), index_col=0)
    Xtr = rd("Xtrain_rawDf.csv")
    Xte = rd("Xtest_rawDf.csv")
    Ytr = rd("YtrainDf.csv")
    Yte = rd("YtestDf.csv")
    Ktr = rd("Xtrain_kernel.csv")
    Kte = rd("Xtest_kernel.csv")
    return Xtr, Xte, Ytr, Yte, Ktr, Kte


def fit_predict_en(Xtr, Ytr, Xte, alpha, l1, min_samples):
    """Per-drug ElasticNet on the (already-scaled) feature matrix, masking NaN labels."""
    pred = np.full((Xte.shape[0], Ytr.shape[1]), np.nan)
    for d in range(Ytr.shape[1]):
        y = Ytr[:, d]
        m = ~np.isnan(y)
        if m.sum() < min_samples:
            continue
        model = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000)
        model.fit(Xtr[m], y[m])
        pred[:, d] = model.predict(Xte)
    return pred


def fit_predict_krr(Ktr, Ytr, Kte, alpha, min_samples):
    """Per-drug Kernel Ridge with the precomputed Pearson kernel, masking NaN labels."""
    pred = np.full((Kte.shape[0], Ytr.shape[1]), np.nan)
    for d in range(Ytr.shape[1]):
        y = Ytr[:, d]
        m = ~np.isnan(y)
        if m.sum() < min_samples:
            continue
        model = KernelRidge(alpha=alpha, kernel="precomputed")
        model.fit(Ktr[np.ix_(m, m)], y[m])
        pred[:, d] = model.predict(Kte[:, m])
    return pred


def split_inner(n, val_frac, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    nval = max(1, int(round(n * val_frac)))
    return idx[nval:], idx[:nval]  # inner_train, val


def tune_en(Xtr, Ytr, val_frac, seed, min_samples, alphas, l1s):
    inner, val = split_inner(Xtr.shape[0], val_frac, seed)
    scaler = StandardScaler().fit(Xtr[inner])
    Xi, Xv = scaler.transform(Xtr[inner]), scaler.transform(Xtr[val])
    k = Ytr.shape[1]
    best = (-np.inf, alphas[0], l1s[0])
    for a in alphas:
        for l1 in l1s:
            pred = fit_predict_en(Xi, Ytr[inner], Xv, a, l1, min_samples)
            score = np.nanmean(NDCGk(Ytr[val], pred, k))
            if score > best[0]:
                best = (score, a, l1)
    return best[1], best[2], best[0]


def tune_krr(Ktr, Ytr, val_frac, seed, min_samples, alphas):
    inner, val = split_inner(Ktr.shape[0], val_frac, seed)
    Kii = Ktr[np.ix_(inner, inner)]
    Kvi = Ktr[np.ix_(val, inner)]
    k = Ytr.shape[1]
    best = (-np.inf, alphas[0])
    for a in alphas:
        pred = fit_predict_krr(Kii, Ytr[inner], Kvi, a, min_samples)
        score = np.nanmean(NDCGk(Ytr[val], pred, k))
        if score > best[0]:
            best = (score, a)
    return best[1], best[0]


def main():
    args = parse_args()
    setup_logger()
    with open(args.config) as f:
        config = yaml.full_load(f)
    logger.info("config=%s | methods=%s | val_frac=%s | seed=%s",
                args.config, args.methods, args.val_frac, args.seed)

    analysis = config["analysis"]
    assert analysis == "FULL", "baselines.py only handles the FULL analysis"
    data_name = config["data"]
    nfolds = config["nfolds"]
    f = config["f"]
    en_alphas = config.get("en_alphas") or EN_ALPHAS
    en_l1s = config.get("en_l1ratios") or EN_L1RATIOS
    # config krr_alphas is often a single placeholder value; widen it if so.
    krr_alphas = config.get("krr_alphas") or KRR_ALPHAS
    if len(krr_alphas) < 2:
        krr_alphas = KRR_ALPHAS

    fold_root = os.path.join(os.getcwd(), data_name, "CV", "FULL")

    for method in args.methods:
        logger.info("=== %s (%d folds, f=%s) ===", method, nfolds, f)
        for i in range(nfolds):
            out = get_result_filename(method, "FULL", data_name, i, f)
            if os.path.exists(out) and not args.overwrite:
                logger.info("Fold %d: %s exists, skipping", i, out)
                continue
            fold_dir = os.path.join(fold_root, "Fold{}".format(i))
            t0 = time.time()
            Xtr_df, Xte_df, Ytr_df, Yte_df, Ktr_df, Kte_df = load_fold(fold_dir)
            Ytr = Ytr_df.values.astype(float)
            Yte = Yte_df.values.astype(float)

            if method == "EN":
                a, l1, vs = tune_en(Xtr_df.values, Ytr, args.val_frac, args.seed,
                                    args.min_samples, en_alphas, en_l1s)
                scaler = StandardScaler().fit(Xtr_df.values)
                Xtr = scaler.transform(Xtr_df.values)
                Xte = scaler.transform(Xte_df.values)
                Ypred = fit_predict_en(Xtr, Ytr, Xte, a, l1, args.min_samples)
                params = "alpha={}, l1_ratio={} (val NDCG={:.4f})".format(a, l1, vs)
            else:  # KRR
                a, vs = tune_krr(Ktr_df.values, Ytr, args.val_frac, args.seed,
                                 args.min_samples, krr_alphas)
                Ypred = fit_predict_krr(Ktr_df.values, Ytr, Kte_df.values, a, args.min_samples)
                params = "alpha={} (val NDCG={:.4f})".format(a, vs)

            np.savez(out, Y_true=Yte, Y_pred=Ypred)
            test_ndcg = np.nanmean(NDCGk(Yte, Ypred, Yte.shape[1]))
            logger.info("Fold %d: %s | full-rank test NDCG=%.4f | %.1fs -> %s",
                        i, params, test_ndcg, time.time() - t0, out)


if __name__ == "__main__":
    main()
