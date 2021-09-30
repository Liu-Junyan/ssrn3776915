# -*- coding: utf-8 -*-
"""
This file is refactored from main.py. It is optimized in many ways. For example, it stores temporary results in memory instead of disk, 
and apply more efficient callback functions, where the lists of coefficients are pre-calculated. For performance reference, it took
me about 7,000 seconds (around 2 hours) to run this file.
The output is store in feature_dict.pkl: Dict[str, pd.DataFrame].
"""

import os
from feature_util import *
import pandas as pd
import numpy as np
import random
from functools import reduce
from constants import FEATURE_SET


def main():
    # mat_dict: Dict[str, pd.DataFrame] = {
    #     f.name[:-4]: pd.read_pickle(f)[["date", "close"]].set_axis(
    #         ["Date", "Price"], axis=1
    #     )
    #     for f in os.scandir("../pkl/")
    # }
    mat_dict: Dict[str, pd.DataFrame] = {}
    for f in os.scandir("../pkl/"):
        df: pd.DataFrame = pd.read_pickle(f)[["date", "close"]].set_axis(
            ["Date", "Price"], axis=1
        )
        df["Date"] = df["Date"].apply(date_s_to_int)
        df["Price"] = df["Price"].astype(float)
        mat_dict[f.name[:-4]] = df  # f.name example: "SH600000.pkl"

    for key in list(mat_dict.keys()):
        mat = mat_dict[key]
        mat = mat[mat["Price"] > 0]
        if not (
            mat["Date"].iloc[0] == 20000104
            and mat["Date"].iloc[-1] == 20210929
            and mat["Date"].unique().size > 5000
        ):
            mat_dict.pop(key)
        else:
            mat_dict[key] = mat

    sp_dict: Dict[str, pd.DataFrame] = {}
    random.seed(20210928)
    sample_keys = random.sample(mat_dict.keys(), 100)
    for key in sample_keys:
        mat = mat_dict[key]
        mat["lnPrice"] = np.log(mat["Price"])
        res = (
            mat.groupby("Date")[["lnPrice"]]
            .agg(realized_variance_d_vec)
            .rename(columns={"lnPrice": "RV^d"})
        )

        res["Open"] = mat.groupby("Date")["lnPrice"].head(1).to_list()
        res["Close"] = mat.groupby("Date")["lnPrice"].tail(1).to_list()
        res["RV^d"] = res["RV^d"] + np.square(
            res["Open"].shift(-1) - res["Close"]
        ).fillna(0)
        res.drop(columns=["Open", "Close"], inplace=True)
        sp_dict[key] = res

    for key in sp_dict.keys():
        res = sp_dict[key]
        for period in Period:
            if period != Period.d:
                res[f"RV^{period.name}"] = res["RV^d"].rolling(period.value).mean()
        sp_dict[key] = res

    theta2_dict = MIDAS_grid_search(sp_dict)

    for key in sp_dict.keys():
        res = sp_dict[key]
        for period in Period:
            theta2 = theta2_dict[key][f"{period.name}"]
            a_list = [
                ((1 - i / 50) ** (theta2 - 1)) * gamma(1 + theta2) / gamma(theta2)
                for i in range(1, 51)
            ]
            res[f"MIDAS^{period.name}"] = (
                res["RV^d"].rolling(50).apply(MIDAS_al, args=(a_list,))
            )

        mat = mat_dict[key]
        df = mat.groupby("Date")["lnPrice"].agg(realized_semivariance_d_vec)
        res[["RVP^d", "RVN^d"]] = df.apply(pd.Series)

        sp_dict[key] = res

    temp_dict = {
        key: mat_dict[key]
        .groupby("Date")[["lnPrice"]]
        .agg(realized_quarticity_d)
        .rename(columns={"lnPrice": "RQ^d"})
        for key in sp_dict.keys()
    }

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

    all_tradedays = reduce(np.union1d, [res.index for res in sp_dict.values()])
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
            1 / RV_sub.shape[1] * np.sum(RV_sub / LM_sub, axis=1) * LM_sub[key]
        )

        CoM = 5
        lmbda = np.log(1 + 1 / CoM)
        exp_list = [np.exp(-i * lmbda) for i in range(1, 501)]
        res["ExpGlRV"] = (
            res["GlRV"].rolling(500).apply(Exp_realized_variance_expl, args=(exp_list,))
        )
        sp_dict[key] = res

    feature_dict = sp_dict
    feature_panel = pd.DataFrame()
    for key in feature_dict.keys():
        feature = feature_dict[key]
        for period in Period:
            feature[f"RV_res^{period.name}"] = (
                feature["RV^d"].rolling(period.value).mean().shift(-period.value)
            )
        feature["Stock"] = key
        feature["Date"] = feature.index.values
        feature_panel = pd.concat([feature_panel, feature], ignore_index=True)
    feature_panel.dropna(subset=FEATURE_SET, inplace=True)
    feature_panel.drop(columns=["GlRV"], inplace=True)
    feature_panel.reset_index(drop=True, inplace=True)
    feature_panel.to_pickle("../feature_panel.pkl")


if __name__ == "__main__":
    main()
