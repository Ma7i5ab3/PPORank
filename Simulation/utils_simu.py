import numpy as np
import pandas as pd


def noramlize_y(Ytrain, Ytest):
    """Z-score normalise Y using train statistics; apply to test."""
    mean = np.nanmean(Ytrain, axis=0)
    std  = np.nanstd(Ytrain, axis=0)
    std[std == 0] = 1.0
    return (Ytrain - mean) / std, (Ytest - mean) / std


def read_simu(data_dir, scenario, analysis, N, P, M, miss_ratio):
    """Load pre-generated simulation CSVs."""
    import os
    dims = f"N{N}_P{P}_M{M}"
    base = os.path.join(data_dir, dims)
    X = pd.read_csv(os.path.join(base, "Xdf.csv"), index_col=0)
    W = pd.read_csv(os.path.join(base, "Wdf.csv"), index_col=0)
    fn = f"Rsparse_miss{miss_ratio}df.csv" if analysis == "sparse" else "Rnoisedf.csv"
    Y = pd.read_csv(os.path.join(base, scenario, fn), index_col=0)
    return X, W, Y
