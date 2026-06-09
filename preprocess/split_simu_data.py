"""
split_simu_data.py
Standalone CV-fold splitter for SimuData (no PyTorch/CaDRRes dependency).
Replicates Split_Simu_Data() from prepare_simu.py.

Run from project root:
  python preprocess/split_simu_data.py --N 1000  --scenarios linear,quad,exp
  python preprocess/split_simu_data.py --N 10000 --scenarios linear,quad,exp
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/SimuData")
    p.add_argument("--N",          type=int,   default=10000)
    p.add_argument("--P",          type=int,   default=1000)
    p.add_argument("--M",          type=int,   default=250)
    p.add_argument("--miss_ratio", type=float, default=0.95)
    p.add_argument("--scenarios",  default="linear,quad,exp")
    p.add_argument("--nfolds",     type=int,   default=5)
    p.add_argument("--seed",       type=int,   default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    dims       = f"N{args.N}_P{args.P}_M{args.M}"
    simu_dir   = os.path.join(args.data_dir, dims)

    X_df = pd.read_csv(os.path.join(simu_dir, "Xdf.csv"), index_col=0)

    kf = KFold(n_splits=args.nfolds, shuffle=True, random_state=args.seed)

    for scenario in args.scenarios.split(","):
        fn = f"Rsparse_miss{args.miss_ratio}df.csv"
        R_fn = os.path.join(simu_dir, scenario, fn)
        Y_df = pd.read_csv(R_fn, index_col=0)

        Ytrain_fn = f"Ytrain_sparse_miss{args.miss_ratio}Df.csv"
        Ytest_fn  = f"Ytest_sparse_miss{args.miss_ratio}Df.csv"

        cv_dir = os.path.join(simu_dir, scenario, "CV")
        for i, (train_idx, test_idx) in enumerate(kf.split(X_df)):
            fold_dir = os.path.join(cv_dir, "sparse", f"Fold{i}")
            os.makedirs(fold_dir, exist_ok=True)

            X_train = X_df.iloc[train_idx]
            X_test  = X_df.iloc[test_idx]
            Y_train = Y_df.iloc[train_idx]
            Y_test  = Y_df.iloc[test_idx]

            X_train.to_csv(os.path.join(fold_dir, "XtrainDf.csv"))
            X_test.to_csv(os.path.join(fold_dir,  "XtestDf.csv"))
            Y_train.to_csv(os.path.join(fold_dir, Ytrain_fn))
            Y_test.to_csv(os.path.join(fold_dir,  Ytest_fn))
            np.savez_compressed(os.path.join(fold_dir, "FulltrainDf.npz"),
                                Xtrain=X_train.values, Ytrain=Y_train.values)
            np.savez_compressed(os.path.join(fold_dir, "FulltestDf.npz"),
                                Xtest=X_test.values,  Ytest=Y_test.values)

        print(f"  {dims}/{scenario}: {args.nfolds} folds written to CV/sparse/")

    print("Done.")


if __name__ == "__main__":
    main()
