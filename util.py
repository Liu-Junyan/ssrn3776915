# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:55:28 2021

@author: John
"""

import pandas as pd
import numpy as np
import pickle
from timeit import default_timer as timer


def filter_by_nrows(filename) -> None:
    '''
    Use the number of observations as a benchmark to filter stocks, and store the valid stocks as pickle.

    Parameters
    ----------
    filename : os.DirEntry
        filename

    Returns
    -------
    None.

    '''
    mat = pd.read_csv(filename, header=None)
    if mat.shape[0] > 100000:
        curr_name = '/project/pkl/' + filename.name[0:8] + '.pkl'
        pickle.dump(mat, open(curr_name, 'wb'))
    return

def realized_variance_d(grouped: pd.core.series.Series) -> float:
    '''
    Calculate annualized daily realized variance.

    Parameters
    ----------
    grouped : pandas.core.series.Series
        Intraday stock price table grouped by date.

    Returns
    -------
    float
        Annualized daily realized variance.

    '''
    
    res = []
    grouped = np.log(grouped)
    
    for i in range(1, len(grouped)):
        res.append(grouped.iloc[i] - grouped.iloc[i-1])

    return 252 * np.sum(np.square(res))

def realized_variance_d_vec(grouped: pd.core.series.Series) -> float:
    '''
    Vectorized variant of realized_variance_d, that runs faster.
    Calculate annualized daily realized variance.

    Parameters
    ----------
    grouped : pandas.core.series.Series
        Intraday stock price table grouped by date.

    Returns
    -------
    float
        Annualized daily realized variance.

    '''
    grouped = np.log(grouped)
    grouped = grouped.diff()[1:]
    return 252 * np.sum(np.square(grouped))

def performance_test():
    mat: pd.core.frame.DataFrame = pickle.load(open('./pkl/SH600000.pkl', 'rb'))
    method = realized_variance_d
    start = timer()
    temp = mat.groupby(0)[[2]].agg(method)
    temp = mat.groupby(0)[[2]].agg(method)
    temp = mat.groupby(0)[[2]].agg(method)
    temp = mat.groupby(0)[[2]].agg(method)
    temp = mat.groupby(0)[[2]].agg(method)
    end = timer()
    print(end - start)