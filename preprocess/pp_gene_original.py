"""
.. module:: gexp
    :synopsis Preprocessing gene expression data

.. moduleauthor:: Nok <suphavilaic@gis.a-star.edu.sg>

"""

import pandas as pd
import numpy as np
from scipy import stats
import time


def log2_exp(exp_df):
    """Calculate log2 gene expression
    """

    return np.log2(exp_df + 1)


def normalize_log2_mean_fc(log2_exp_df):
    """Calculate gene expression foldchange based on median of each genes. The sample size should be large enough (>10).
    """

    return (log2_exp_df.T - log2_exp_df.mean(axis=1)).T, pd.DataFrame(log2_exp_df.mean(axis=1), columns=['median'])


def normalize_log2_mean_fc_with_ref(log2_exp_df, log2_ref_exp_df):

    common_genes = set(log2_ref_exp_df.index).intersection(log2_exp_df.index)
    log2_exp_df = log2_exp_df.loc[common_genes]
    log2_ref_exp_df = log2_ref_exp_df.loc[common_genes]

    return (log2_exp_df.T - log2_ref_exp_df.mean(axis=1)).T, pd.DataFrame(log2_ref_exp_df.mean(axis=1), columns=['median'])


def normalize_L1000_suite():
    """
    """

# TODO: make this multiprocessor


def calculate_kernel_feature(log2_median_fc_exp_df, ref_log2_median_fc_exp_df, gene_list):
    common_genes = [g for g in gene_list if (g in log2_median_fc_exp_df.index)
                    and (g in ref_log2_median_fc_exp_df.index)]

    print('Calculating kernel features based on', len(common_genes), 'common genes')

    print(log2_median_fc_exp_df.shape, ref_log2_median_fc_exp_df.shape)

    sample_list = list(log2_median_fc_exp_df.columns)
    ref_sample_list = list(ref_log2_median_fc_exp_df.columns)

    exp_mat = np.array(log2_median_fc_exp_df.loc[common_genes], dtype='float')
    ref_exp_mat = np.array(ref_log2_median_fc_exp_df.loc[common_genes], dtype='float')

    # Vectorized Pearson correlation: z-score each column, then dot product / n_genes
    def _zscore(m):
        mu = m.mean(axis=0, keepdims=True)
        sd = m.std(axis=0, keepdims=True)
        sd[sd == 0] = 1
        return (m - mu) / sd

    n_genes = exp_mat.shape[0]
    sim_mat = (_zscore(exp_mat).T @ _zscore(ref_exp_mat)) / n_genes

    return pd.DataFrame(sim_mat, columns=ref_sample_list, index=sample_list)


def get_gene_list(gene_list_fname):
    return list(pd.read_csv(gene_list_fname, header=None)[0].values)
