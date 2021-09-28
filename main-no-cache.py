# -*- coding: utf-8 -*-
"""
This file is refactored from main.py. It is optimized in many ways. For example, it stores temporary results in memory instead of disk, 
and it adopts more efficient callback functions, where the lists of coefficients are pre-calculated. For performance reference, it took
me 7,000 seconds (about 2 hours) to run this file.
The output is store in feature_dict.pkl: Dict[str, pd.DataFrame].
"""

import os
from typing import Dict
from util import *
import pickle
import pandas as pd
import numpy as np
import random


def main():
    mat_dict: Dict[str, pd.DataFrame] = {
        f.name[:-4]: pd.read_pickle(f) for f in os.scandir("./pkl/")
    }
    res_dict: Dict[str, pd.DataFrame] = {
        key: mat_dict[key].groupby(0)[[2]].agg(realized_variance_d_vec)
        for key in mat_dict.keys()
    }
    for key in list(res_dict.keys()):
        res = res_dict[key]
        if not (
            res.index[0] == 20050104 and res.index[-1] == 20161230 and len(res) >= 2800
        ):
            res_dict.pop(key)

    sample = random.sample(res_dict.keys(), 100)
    sp_dict: Dict[str, pd.DataFrame] = {key: res_dict[key] for key in sample}

    for key in sp_dict.keys():
        res = sp_dict[key]
        res.rename(columns={2: "RV^d"}, inplace=True)
        for period in Period:
            if period != Period.d:
                res[f"RV^{period.name}"] = res["RV^d"].rolling(period.value).mean()
        sp_dict[key] = res

    (theta2_d, theta2_w, theta2_m, theta2_q) = MIDAS_grid_search_nocache(sp_dict)
    print(theta2_d, theta2_w, theta2_m, theta2_q)
    theta2_dict = {
        "theta2_d": theta2_d,
        "theta2_w": theta2_w,
        "theta2_m": theta2_m,
        "theta2_q": theta2_q,
    }

    for key in sp_dict.keys():
        res = sp_dict[key]
        for period in Period:
            theta2 = theta2_dict[f"theta2_{period.name}"]
            a_list = [
                ((1 - i / 50) ** (theta2 - 1)) * gamma(1 + theta2) / gamma(theta2)
                for i in range(1, 51)
            ]
            res[f"MIDAS^{period.name}"] = (
                res["RV^d"].rolling(50).apply(MIDAS_al, args=(a_list,))
            )

        mat = mat_dict[key]
        df = mat.groupby(0)[2].agg(realized_semivariance_d_vec)
        df = df.apply(pd.Series)
        res[["RVP^d", "RVN^d"]] = df

        sp_dict[key] = res

    temp_dict = {}
    for key in sp_dict.keys():
        mat = mat_dict[key]
        df = mat.groupby(0)[[2]].agg(realized_quarticity_d)
        df.rename(columns={2: "RQ^d"}, inplace=True)
        temp_dict[key] = df

    for key in temp_dict.keys():
        df = temp_dict[key]
        for period in Period:
            df[f"RQ^{period.name}"] = df["RQ^d"].rolling(period.value).mean()
        df = np.sqrt(df)
        res = sp_dict[key]
        for period in Period:
            res[f"HARQ^{period.name}"] = (
                df[f"RQ^{period.name}"] * res[f"RV^{period.name}"]
            )
        sp_dict[key] = res

    for key in sp_dict.keys():
        res = sp_dict[key]
        for CoM in (1, 5, 25, 125):
            lmbda = np.log(1 + 1 / CoM)
            exp_list = [np.exp(-i * lmbda) for i in range(1, 501)]
            res[f"ExpRV^{CoM}"] = (
                res["RV^d"]
                .rolling(500)
                .apply(Exp_realized_variance_expl, args=(exp_list,))
            )
        sp_dict[key] = res

    all_tradedays = []
    for key in sp_dict.keys():
        res = sp_dict[key]
        all_tradedays = np.union1d(all_tradedays, res.index)
    RV_all = pd.DataFrame(index=all_tradedays)
    for key in sp_dict.keys():
        res = sp_dict[key]
        RV_all[key] = res["RV^d"]

    LM_all = RV_all
    for i in range(RV_all.shape[0]):
        LM_all.iloc[i] = np.mean(RV_all.iloc[: i + 1])
    RV_all = RV_all.interpolate()

    for key in sp_dict.keys():
        res = sp_dict[key]
        RV_sub = RV_all.loc[res.index]
        LM_sub = LM_all.loc[res.index]
        res["GlRV"] = (
            1 / RV_all.shape[1] * np.sum(RV_sub / LM_sub, axis=1) * LM_sub[key]
        )

        CoM = 5
        lmbda = np.log(1 + 1 / CoM)
        exp_list = [np.exp(-i * lmbda) for i in range(1, 501)]
        res["ExpGlRV"] = (
            res["GlRV"].rolling(500).apply(Exp_realized_variance_expl, args=(exp_list,))
        )

        sp_dict[key] = res

    pickle.dump(sp_dict, open("feature_dict.pkl", "wb"))


if __name__ == "__main__":
    main()
