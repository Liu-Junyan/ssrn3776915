# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:55:28 2021

@author: John
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import pickle
from timeit import default_timer as timer
import os
from scipy.special import gamma
from sklearn.linear_model import LinearRegression
from constant import Period
from sys import exit


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
        curr_name = './pkl/' + filename.name[:-4] + '.pkl'
        mat.to_pickle(curr_name)

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

def realized_variance_d_vec(s: pd.Series) -> float:
    '''
    Vectorized variant of realized_variance_d, that runs faster.
    Calculate annualized daily realized variance.

    Parameters
    ----------
    s : pandas.Series
        Intraday stock price series in one day.

    Returns
    -------
    float
        Annualized daily realized variance.

    '''
    s = np.log(s)
    s = s.diff()[1:]
    return 252 * np.sum(np.square(s))

def realized_semivariance_d_vec(s: pd.Series) -> Tuple[float, float]:
    '''
    Calculate annualized daily realized semivariance.

    Parameters
    ----------
    s : pandas.Series
        Intraday stock price series in one day.

    Returns
    -------
    Tuple(float, float)
        (RVP^d, RVN^d). Annualized daily realized semivariance.

    '''
    s = np.log(s)
    s = s.diff()[1:]
    s_positive: pd.Series = s.apply(max, args=(0,))
    s_negative: pd.Series = s - s_positive
    return (252 * np.sum(np.square(s_positive)), 252 * np.sum(np.square(s_negative)))

def realized_quarticity_d(s: pd.Series) -> float:
    s = np.log(s)
    s = s.diff()[1:]
    n = s.size
    return (252 ** 2) * n / 3 * np.sum(np.power(s, 4))

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

def MIDAS_grid_search() -> Tuple[int, int, int, int]:
    '''
    Brute force to find the optimal theta2s for MIDAS algorithm.

    Returns
    -------
    Tuple(int, int, int, int)
        A tuple of optimal theta2s for MIDAS_d, MIDAS_w, MIDAS_m, and MIDAS_q, which maximize R^2s.

    '''
    lm: LinearRegression = LinearRegression()
    theta2_profile: pd.DataFrame = pd.DataFrame(index=list(range(1,100)), columns=[period.name for period in Period], dtype=float)
    for theta2 in range(1, 100):
        RSS_dict = {period.name:0 for period in Period}
        TSS_dict = {period.name:0 for period in Period}
        for filename in os.scandir('./sp/'):
            res: pd.DataFrame = pd.read_pickle(filename.path)
            MIDAS_series: pd.Series = res['RV^d'].rolling(50).apply(MIDAS, args=(theta2,))
            for period in Period: # Search for all periods at once
                df: pd.DataFrame = pd.DataFrame({
                    'RV': res[f'RV^{period.name}'],
                    'MIDAS': MIDAS_series.shift(period.value)
                })
                df.dropna(inplace=True)
                lm.fit(df[['MIDAS']], df['RV'])
                df['predicted'] = lm.predict(df[['MIDAS']])
                RSS_dict[period.name] += np.sum(np.square(df['RV'] - df['predicted']))
                TSS_dict[period.name] += np.sum(np.square(df['RV'] - np.mean(df['RV'])))
        for period in Period:
            theta2_profile.loc[theta2, period.name] = 1 - RSS_dict[period.name] / TSS_dict[period.name]
        pass
    theta2_profile.to_pickle('profile.pkl')
    return (theta2_profile['d'].idxmax(), theta2_profile['w'].idxmax(), theta2_profile['m'].idxmax(), theta2_profile['q'].idxmax())

def MIDAS_grid_search_nocache(sp_dict: Dict[str, pd.DataFrame]) -> Tuple[int, int, int, int]:
    '''
    Brute force to find the optimal theta2s for MIDAS algorithm.

    Returns
    -------
    Tuple(int, int, int, int)
        A tuple of optimal theta2s for MIDAS_d, MIDAS_w, MIDAS_m, and MIDAS_q, which maximize R^2s.

    '''
    lm: LinearRegression = LinearRegression()
    theta2_profile: pd.DataFrame = pd.DataFrame(index=list(range(1,100)), columns=[period.name for period in Period], dtype=float)
    for theta2 in range(1, 100):
        RSS_dict = {period.name:0 for period in Period}
        TSS_dict = {period.name:0 for period in Period}
        for key in sp_dict.keys():
            res: pd.DataFrame = sp_dict[key]
            MIDAS_series: pd.Series = res['RV^d'].rolling(50).apply(MIDAS, args=(theta2,))
            for period in Period: # Search for all periods at once
                df: pd.DataFrame = pd.DataFrame({
                    'RV': res[f'RV^{period.name}'],
                    'MIDAS': MIDAS_series.shift(period.value)
                })
                df.dropna(inplace=True)
                lm.fit(df[['MIDAS']], df['RV'])
                df['predicted'] = lm.predict(df[['MIDAS']])
                RSS_dict[period.name] += np.sum(np.square(df['RV'] - df['predicted']))
                TSS_dict[period.name] += np.sum(np.square(df['RV'] - np.mean(df['RV'])))
        for period in Period:
            theta2_profile.loc[theta2, period.name] = 1 - RSS_dict[period.name] / TSS_dict[period.name]
        pass
    theta2_profile.to_pickle('profile.pkl')
    return (theta2_profile['d'].idxmax(), theta2_profile['w'].idxmax(), theta2_profile['m'].idxmax(), theta2_profile['q'].idxmax())

def Exp_realized_variance(s: pd.Series, CoM: int) -> float:
    lmbda = np.log(1 + 1 / CoM)
    exp_list = [np.exp(-i * lmbda) for i in range(1, 501)]
    return np.dot(s, exp_list[::-1]) / np.sum(exp_list)

def Exp_realized_variance_expl(s: pd.Series, exp_list: List[float]) -> float:
    return np.dot(s, exp_list[::-1]) / np.sum(exp_list)
    
# def performance_test():
#     mat: pd.core.frame.DataFrame = pickle.load(open('./pkl/SH600000.pkl', 'rb'))
#     method = realized_variance_d
#     start = timer()
#     temp = mat.groupby(0)[[2]].agg(method)
#     temp = mat.groupby(0)[[2]].agg(method)
#     temp = mat.groupby(0)[[2]].agg(method)
#     temp = mat.groupby(0)[[2]].agg(method)
#     temp = mat.groupby(0)[[2]].agg(method)
#     end = timer()
#     print(end - start)