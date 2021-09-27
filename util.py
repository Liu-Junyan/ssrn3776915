# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:55:28 2021

@author: John
"""

import pandas as pd
import numpy as np
import pickle
from timeit import default_timer as timer
import os
from scipy.special import gamma
from sklearn.linear_model import LinearRegression


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

def filter_by_ndates() -> None:
    '''
    Use the number of dates as a benchmark to filter stocks, and delete invalid pickles. 
    Valid stocks have a record starting at 2005.1.4, ending at 2016.12.30, and have more than 2800 entries.

    Returns
    -------
    None.

    '''
    for filename in os.scandir('./res/'):
        res: pd.core.frame.DataFrame = pickle.load(open(filename, 'rb'))
        if not (res.index[0] == 20050104 and res.index[-1] == 20161230 and len(res) >= 2800):
            os.remove(filename.path)
            print(f'{filename.name} deleted.')

def realized_variance_d(grouped: pd.Series) -> float:
    '''
    Calculate annualized daily realized variance.

    Parameters
    ----------
    grouped : pandas.Series
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

def realized_variance_d_vec(grouped: pd.Series) -> float:
    '''
    Vectorized variant of realized_variance_d, that runs faster.
    Calculate annualized daily realized variance.

    Parameters
    ----------
    grouped : pandas.Series
        Intraday stock price table grouped by date.

    Returns
    -------
    float
        Annualized daily realized variance.

    '''
    grouped = np.log(grouped)
    grouped = grouped.diff()[1:]
    return 252 * np.sum(np.square(grouped))

def rolling_mean(s: pd.Series, N: int) -> pd.Series:
    '''
    Calculate the rolling mean with window size N of a series.
    Use this function to calculate RV^w, RV^m and RV^q.

    Parameters
    ----------
    s : pandas.Series
        A series of RV^d.
    N : int
        Window size.

    Returns
    -------
    pandas.Series
        Rolling mean of RV^d.

    '''
    return s.rolling(N).mean()

def MIDAS(s: pd.Series, theta2: int = 1) -> float:
    '''
    Calculate MIDAS from a series of 50 RV^d.

    Parameters
    ----------
    s : pd.Series
        A series of 50 annualized realized variance.
    theta2 : int
        Hyperparameter of MIDAS algorithm. Need to perform a grid search to find the optimal value.

    Returns
    -------
    float
        MIDAS.

    '''
    a_list = []
    for index in range(50):
        i = index + 1
        a = ((1 - i/50) ** (theta2 - 1)) * gamma(1 + theta2) / gamma(theta2)
        a_list.append(a)
    MIDAS = np.dot(s, a_list[::-1]) / np.sum(a_list)
    return MIDAS

def MIDAS_grid_search() -> (float, float, float, float):
    '''
    Find the optimal theta2s for MIDAS algorithm.

    Returns
    -------
    (float, float, float, float)
        Optimal theta2 for MIDAS_d, MIDAS_w, MIDAS_m, and MIDAS_q.

    '''
    lm: LinearRegression = LinearRegression()
    
    for theta2 in range(1, 100):
        for filename in os.scandir('./sp/'):
            res: pd.DataFrame = pd.read_pickle(filename.path)
            MIDAS_series: pd.Series = res['RV^d'].rolling(50).apply(MIDAS, args=(theta2,))
            # k = d
            df_d = pd.DataFrame({
                'RV^d': res['RV^d'],
                'MIDAS': MIDAS_series.shift(1)
            })
            lm.fit(df_d[['MIDAS']], df_d['RV^d'])
    pass

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