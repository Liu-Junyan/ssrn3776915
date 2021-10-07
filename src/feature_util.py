# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:55:28 2021

@author: John
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import gamma
from sklearn.linear_model import LinearRegression

from constants import Period


def date_s_to_int(date_s: str) -> int:
    """Convert string date to integer.

    Args:
        date_s (str): String date of format YYYY-MM-DD.

    Returns:
        int: Integer date.
    """
    multipier = 10000000
    date_int = 0
    for c in date_s:
        if c != "-":
            date_int += int(c) * multipier
            multipier //= 10
    return date_int


def realized_variance_d(s: pd.Series) -> float:
    """Calculate annualized daily realized variance.

    Args:
        s (pd.Series): Intraday log stock price series in one day.

    Returns:
        float: Annualized daily realized variance.
    """
    res = []
    for i in range(1, len(s)):
        res.append(s.iloc[i] - s.iloc[i - 1])

    return 252 * np.sum(np.square(res))


def realized_variance_d_vec(s: pd.Series) -> float:
    """Vectorized variant of realized_variance_d, that runs faster.
    Calculate annualized daily realized variance.

    Args:
        s (pd.Series): Intraday log stock price series in one day.

    Returns:
        float: Annualized daily realized variance.
    """
    s = s.diff()[1:]
    return 252 * np.sum(np.square(s))


def realized_semivariance_d_vec(s: pd.Series) -> Tuple[float, float]:
    """Calculate annualized daily realized semivariances.

    Args:
        s (pd.Series): Intraday log stock price series in one day.

    Returns:
        Tuple[float, float]: (RVP^d, RVN^d). Annualized daily realized semivariances.
    """
    s = s.diff()[1:]
    s_positive: pd.Series = s.apply(max, args=(0,))
    s_negative: pd.Series = s - s_positive
    return 252 * np.sum(np.square(s_positive)), 252 * np.sum(np.square(s_negative))


def realized_quarticity_d(s: pd.Series) -> float:
    """Calculate annualized daily realized quarticity.

    Args:
        s (pd.Series): Intraday log stock price series in one day.

    Returns:
        float: Annualized daily realized quarticity.
    """
    s = s.diff()[1:]
    n = s.size
    return (252 ** 2) * n / 3 * np.sum(np.power(s, 4))


def MIDAS(s: pd.Series, theta2: int = 1) -> float:
    """Calculate MIDAS from a series of 50 RV^d.

    Args:
        s (pd.Series): A series of 50 annualized realized variance.
        theta2 (int, optional): Hyperparameter of MIDAS algorithm. Need to perform a grid search to find the optimal value. Defaults to 1.

    Returns:
        float: MIDAS.
    """
    a_list = []
    for index in range(50):
        i = index + 1
        a = ((1 - i / 50) ** (theta2 - 1)) * gamma(1 + theta2) / gamma(theta2)
        a_list.append(a)
    return np.dot(s, a_list[::-1]) / np.sum(a_list)


def MIDAS_al(s: pd.Series, a_list: List[float]) -> float:
    """A variant of MIDAS(), that accepts pre-calculated a_list to prevent recalculation. Should use this over MIDAS().

    Args:
        s (pd.Series): A series of 50 annualized realized variance.
        a_list (List[float]): A list of precalculated a coefficients of MIDAS.

    Returns:
        float: MIDAS.
    """
    MIDAS = np.dot(s, a_list[::-1]) / np.sum(a_list)
    return MIDAS


def MIDAS_grid_search(sp_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, int]]:
    """Brute-force search the optimal theta2s for MIDAS algorithm. Use with feature.py.

    Returns:
        Tuple[int, int, int, int]: A tuple of optimal theta2s for MIDAS_d, MIDAS_w, MIDAS_m, and MIDAS_q that maximize R^2s.
    """
    lm = LinearRegression()
    theta2_dict: Dict[str, Dict[str, Dict]] = {
        key: pd.Series(index=[period.name for period in Period], dtype=int)
        for key in sp_dict.keys()
    }
    theta2_R2_profile: Dict[str, pd.DataFrame] = {
        key: pd.DataFrame(
            index=list(range(1, 41)),
            columns=[period.name for period in Period],
            dtype=float,
        )
        for key in sp_dict.keys()
    }

    for theta2 in range(1, 41):
        print(f"Processing theta2 = {theta2}")
        a_list = [
            ((1 - i / 50) ** (theta2 - 1)) * gamma(1 + theta2) / gamma(theta2)
            for i in range(1, 51)
        ]

        for key in sp_dict.keys():
            res: pd.DataFrame = sp_dict[key]
            MIDAS_series: pd.Series = res["RV^d"].rolling(50).apply(
                MIDAS_al, args=(a_list,)
            )
            for period in Period:
                df = pd.DataFrame()
                df["RV"] = res[f"RV^{period.name}"]
                df["MIDAS"] = MIDAS_series.shift(period.value)
                df.dropna(inplace=True)
                lm.fit(df[["MIDAS"]], df["RV"])
                df["predicted"] = lm.predict(df[["MIDAS"]])
                RSS = np.sum(np.square(df["RV"] - df["predicted"]))
                TSS = np.sum(np.square(df["RV"] - np.mean(df["RV"])))
                theta2_R2_profile[key].loc[theta2, period.name] = 1 - RSS / TSS

    for key in sp_dict.keys():
        theta2_dict[key] = {
            period.name: theta2_R2_profile[key][period.name].idxmax()
            for period in Period
        }

    return theta2_dict


def Exp_realized_variance(s: pd.Series, CoM: int) -> float:
    """Calculate exponential realized variance.

    Args:
        s (pd.Series): A series of 500 annualized realized variance.
        CoM (int): Center-of-mass.

    Returns:
        float: Exponential realized variance.
    """
    lmbda = np.log(1 + 1 / CoM)
    exp_list = [np.exp(-i * lmbda) for i in range(1, 501)]
    return np.dot(s, exp_list[::-1]) / np.sum(exp_list)


def Exp_realized_variance_expl(s: pd.Series, exp_list: List[float]) -> float:
    """A variant of Exp_realized_variance(), that accepts pre-calculated exp_list to prevent recalculation. Should use this over Exp_realized_variance().

    Args:
        s (pd.Series): A series of 500 annualized realized variance.
        exp_list (List[float]): A list of precalculated exp coefficients of EWMA.

    Returns:
        float: Exponential realized variance.
    """
    return np.dot(s, exp_list[::-1]) / np.sum(exp_list)
