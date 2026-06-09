"""
generate_simu_data.py
Generates synthetic drug-response data for PPORank simulation experiments
(Section 3.5 of Liu et al., Statistics in Medicine 2022).

Scenarios
---------
  linear : Y = 0.2  * X @ W + ε
  quad   : Y = 0.15 * X**3 @ W + 0.15 * X @ W + ε
  exp    : Y = 0.1  * exp(X) @ W + 0.1 * X**3 @ W + ε

Output structure (mirrors what prepare_simu.py expects):
  data/SimuData/N{N}_P{P}_M{M}/
    Xdf.csv                              # N×P cell-line features
    Wdf.csv                              # P×M drug weight matrix
    {scenario}/
      Rsparse_miss{miss_ratio}df.csv     # N×M IC50, 95% entries masked NaN

Run from project root:
  python preprocess/generate_simu_data.py
  python preprocess/generate_simu_data.py --N 1000 --P 1000 --M 250
"""

import os
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/SimuData")
    p.add_argument("--N",          type=int,   default=10000, help="cell lines")
    p.add_argument("--P",          type=int,   default=1000,  help="features")
    p.add_argument("--M",          type=int,   default=250,   help="drugs")
    p.add_argument("--miss_ratio", type=float, default=0.95,  help="fraction of Y set to NaN")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--scenarios",  default="linear,quad,exp")
    p.add_argument("--noise_std",  type=float, default=0.1,   help="Gaussian noise std")
    return p.parse_args()


def simulate_Y(X, W, scenario, noise_std, rng):
    """Return Y matrix (N×M) for the given scenario."""
    eps = rng.normal(0, noise_std, size=(X.shape[0], W.shape[1]))
    if scenario == "linear":
        return 0.2 * X @ W + eps
    elif scenario == "quad":
        return 0.15 * (X ** 3) @ W + 0.15 * X @ W + eps
    elif scenario == "exp":
        return 0.1 * np.exp(X) @ W + 0.1 * (X ** 3) @ W + eps
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    dims    = f"N{args.N}_P{args.P}_M{args.M}"
    out_dir = os.path.join(args.data_dir, dims)
    os.makedirs(out_dir, exist_ok=True)

    cell_ids = [f"CL{i}" for i in range(args.N)]
    feat_ids = [f"F{j}"  for j in range(args.P)]
    drug_ids = [f"D{k}"  for k in range(args.M)]

    # ── X: standardised normal features ──────────────────────────────────────
    X_fn = os.path.join(out_dir, "Xdf.csv")
    if not os.path.exists(X_fn):
        X_raw = rng.standard_normal((args.N, args.P)).astype(np.float32)
        X_raw = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-8)
        pd.DataFrame(X_raw, index=cell_ids, columns=feat_ids).to_csv(X_fn)
        print(f"  Saved Xdf.csv  ({args.N}×{args.P})")
    else:
        X_raw = pd.read_csv(X_fn, index_col=0).values.astype(np.float32)
        print(f"  Xdf.csv already exists — loaded.")

    # ── W: sparse drug weight matrix (10% non-zero) ───────────────────────────
    W_fn = os.path.join(out_dir, "Wdf.csv")
    if not os.path.exists(W_fn):
        W_raw  = rng.standard_normal((args.P, args.M)).astype(np.float32)
        mask_W = rng.random((args.P, args.M)) < 0.10
        W_raw  = W_raw * mask_W
        pd.DataFrame(W_raw, index=feat_ids, columns=drug_ids).to_csv(W_fn)
        print(f"  Saved Wdf.csv  ({args.P}×{args.M})")
    else:
        W_raw = pd.read_csv(W_fn, index_col=0).values.astype(np.float32)
        print(f"  Wdf.csv already exists — loaded.")

    # ── Y per scenario ────────────────────────────────────────────────────────
    for scenario in args.scenarios.split(","):
        scen_dir = os.path.join(out_dir, scenario)
        os.makedirs(scen_dir, exist_ok=True)

        sparse_fn = os.path.join(scen_dir,
                                 f"Rsparse_miss{args.miss_ratio}df.csv")
        if os.path.exists(sparse_fn):
            print(f"  {scenario}/Rsparse already exists — skipping.")
            continue

        Y = simulate_Y(X_raw, W_raw, scenario, args.noise_std, rng)

        # apply sparsity mask
        mask = rng.random(Y.shape) < args.miss_ratio
        Y_sparse = Y.astype(float)
        Y_sparse[mask] = np.nan

        pd.DataFrame(Y_sparse, index=cell_ids, columns=drug_ids).to_csv(sparse_fn)
        obs = int((~np.isnan(Y_sparse)).sum())
        print(f"  {scenario}: Y range [{Y.min():.2f}, {Y.max():.2f}]  "
              f"observed entries: {obs}/{args.N*args.M} "
              f"({100*obs/(args.N*args.M):.1f}%)")

    print(f"\nDone. Data in: {out_dir}")


if __name__ == "__main__":
    main()
