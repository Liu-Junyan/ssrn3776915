# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:55:28 2021

@author: John
"""

import pandas as pd
import numpy as np
import pickle


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

    return np.sum(np.square(res)) * 252

def realized_variance_d_1(grouped: pd.core.series.Series) -> float:
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
    grouped = np.log(grouped)
    grouped_head = pd.concat([pd.Series([0]), grouped], ignore_index=True)
    grouped_tail = pd.concat([grouped, pd.Series([0])], ignore_index=True)
    grouped = (grouped_tail - grouped_head)[1:-1]
    grouped = 252 * np.sum(np.square(grouped))
    return grouped

'''
for filename in os.scandir('/project/data/'):
    filter_by_nrows(filename)
'''