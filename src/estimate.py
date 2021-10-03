"""
Usage: cd src && python3 estimate.py
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import ensemble
from constants import Period, FEATURE_SET_ALL, T_START, T_END
from estimate_util import *
import pickle
import random

feature_set_dict: Dict[str, List[str]] = {
    "HAR": ["RV^d", "RV^w", "RV^m", "RV^q"],
    "SHAR": ["RVP^d", "RVN^d", "RV^w", "RV^m", "RV^q"],
    "HARQ": ["RV^d", "RV^w", "RV^m", "RV^q", "HARQ^d", "HARQ^w", "HARQ^m", "HARQ^q"],
    "HExpGl": ["ExpRV^1", "ExpRV^5", "ExpRV^25", "ExpRV^125", "ExpGlRV"],
    "All": FEATURE_SET_ALL,
}


def estimate_OLS_Based(
    fp: pd.DataFrame, estimated_dict: Dict[str, pd.DataFrame], predicted: str
):
    """Estimate by an OLS based model.

    Args:
        fp (pd.DataFrame): A panel of features and response variables.
        estimated_dict (Dict[str, pd.DataFrame]): A dict of estimated results. MUTATED.
    """
    lm = linear_model.LinearRegression()
    predicted_var = f"RV_{predicted}"
    feature_set = feature_set_dict[predicted]
    expand_estimated_dict(estimated_dict, predicted_var)
    for period in Period:
        response = f"RV_res^{period.name}"
        estimated = estimated_dict[period.name]
        # t for testing set
        for t in range(T_START, T_END):
            training_panel = fp[fp["Year"] < t]
            testing_panel = fp[fp["Year"] == t]
            training_p_v = validate_panel(training_panel, response)
            testing_p_v = validate_panel(testing_panel, response)
            lm.fit(training_p_v[feature_set], training_p_v[response])
            estimated.loc[estimated["Year"] == t, predicted_var] = lm.predict(
                testing_p_v[feature_set]
            )


def estimate_MIDAS(
    fp: pd.DataFrame, estimated_dict: Dict[str, pd.DataFrame], period: Period
):
    lm = linear_model.LinearRegression()
    predicted_var = "RV_MIDAS"
    feature_set = [f"MIDAS^{period.name}"]
    expand_estimated_dict(estimated_dict, predicted_var)

    response = f"RV_res^{period.name}"
    estimated = estimated_dict[period.name]
    for t in range(T_START, T_END):
        training_panel = fp[fp["Year"] < t]
        testing_panel = fp[fp["Year"] == t]
        training_p_v = validate_panel(training_panel, response)
        testing_p_v = validate_panel(testing_panel, response)
        lm.fit(training_p_v[feature_set], training_p_v[response])
        estimated.loc[estimated["Year"] == t, predicted_var] = lm.predict(
            testing_p_v[feature_set]
        )


def estimate_LASSO(
    fp: pd.DataFrame, estimated_dict: Dict[str, pd.DataFrame], use_stash: bool = False
):
    lasso = linear_model.Lasso()
    lm = linear_model.LinearRegression()
    predicted = "RV_LASSO"
    expand_estimated_dict(estimated_dict, predicted)
    lmbda_table = pd.DataFrame(
        index=range(T_START, T_END - 1),
        columns=[period.name for period in Period],
        dtype=float,
    )
    if use_stash:
        lmbda_table_stash: pd.DataFrame = pickle.load(
            open("../lmbda_table_stash.pkl", "rb")
        )
    # Under training-validation-testing scheme, t = 2008...2015. (testing: 2007...t-1; validation: t; testing: t+1)
    # t for validation set
    for t in range(T_START, T_END - 1):
        training_panel = fp[fp["Year"] < t].copy()
        validation_panel = fp[fp["Year"] == t].copy()
        testing_panel = fp[fp["Year"] == t + 1].copy()

        training_mean = training_panel[FEATURE_SET_ALL].mean()
        training_std = training_panel[FEATURE_SET_ALL].std()

        training_panel[FEATURE_SET_ALL] = standardize(
            training_panel[FEATURE_SET_ALL], training_mean, training_std
        )
        validation_panel[FEATURE_SET_ALL] = standardize(
            validation_panel[FEATURE_SET_ALL], training_mean, training_std
        )
        testing_panel[FEATURE_SET_ALL] = standardize(
            testing_panel[FEATURE_SET_ALL], training_mean, training_std
        )

        for period in Period:
            response = f"RV_res^{period.name}"
            training_p_v = validate_panel(training_panel, response)
            validation_p_v = validate_panel(validation_panel, response)
            testing_p_v = validate_panel(testing_panel, response)

            estimated_copy = estimated_dict[period.name].copy()
            estimated_copy = estimated_copy[estimated_copy["Year"] == t]

            lmbda = (
                lasso_grid_search(training_p_v, validation_p_v, estimated_copy, period)
                if not use_stash
                else lmbda_table_stash.loc[t, period.name]
            )
            lmbda_table.loc[t, period.name] = lmbda
            print(f"LASSO: lambda for year {t} series {period.name} is {lmbda}")
            estimated = estimated_dict[period.name]
            if lmbda != 0:
                lasso.alpha = lmbda
                lasso.fit(training_p_v[FEATURE_SET_ALL], training_p_v[response])
                estimated.loc[estimated["Year"] == t + 1, predicted] = lasso.predict(
                    testing_p_v[FEATURE_SET_ALL]
                )
            else:  # Collapse to OLS
                lm.fit(training_p_v[FEATURE_SET_ALL], training_p_v[response])
                estimated.loc[estimated["Year"] == t + 1, predicted] = lm.predict(
                    testing_p_v[FEATURE_SET_ALL]
                )
    lmbda_table.to_pickle("../lmbda_table_stash.pkl")


def estimate_PCR(
    fp: pd.DataFrame, estimated_dict: Dict[str, pd.DataFrame], use_stash=False
):
    pca = decomposition.PCA()
    lm = linear_model.LinearRegression()
    predicted = "RV_PCR"
    expand_estimated_dict(estimated_dict, predicted)
    pca_n_table = pd.DataFrame(
        index=range(T_START, T_END - 1),
        columns=[period.name for period in Period],
        dtype=int,
    )
    if use_stash:
        pca_n_table_stash: pd.DataFrame = pickle.load(
            open("../pca_n_table_stash.pkl", "rb")
        )
    for t in range(T_START, T_END - 1):
        training_panel = fp[fp["Year"] < t].copy()
        validation_panel = fp[fp["Year"] == t].copy()
        testing_panel = fp[fp["Year"] == t + 1].copy()

        training_mean = training_panel[FEATURE_SET_ALL].mean()
        training_std = training_panel[FEATURE_SET_ALL].std()

        training_panel[FEATURE_SET_ALL] = standardize(
            training_panel[FEATURE_SET_ALL], training_mean, training_std
        )
        validation_panel[FEATURE_SET_ALL] = standardize(
            validation_panel[FEATURE_SET_ALL], training_mean, training_std
        )
        testing_panel[FEATURE_SET_ALL] = standardize(
            testing_panel[FEATURE_SET_ALL], training_mean, training_std
        )

        for period in Period:
            response = f"RV_res^{period.name}"
            training_p_v = validate_panel(training_panel, response)
            validation_p_v = validate_panel(validation_panel, response)
            testing_p_v = validate_panel(testing_panel, response)

            pca_n = (
                pca_grid_search(training_p_v, validation_p_v, period)
                if not use_stash
                else pca_n_table_stash.loc[t, period.name]
            )
            pca_n_table.loc[t, period.name] = pca_n
            print(f"PCR: n for year {t} series {period.name} is {pca_n}")
            estimated = estimated_dict[period.name]
            pca.n_components = pca_n
            pca.fit(training_p_v[FEATURE_SET_ALL])
            training_X = pca.transform(training_p_v[FEATURE_SET_ALL])
            testing_X = pca.transform(testing_p_v[FEATURE_SET_ALL])
            lm.fit(training_X, training_p_v[response])
            estimated.loc[estimated["Year"] == t + 1, predicted] = lm.predict(testing_X)
    pca_n_table.to_pickle("../pca_n_table_stash.pkl")


def estimate_RF(fp: pd.DataFrame, estimated_dict: Dict[str, pd.DataFrame]):
    random.seed(20211002)
    predicted = "RV_RF"
    expand_estimated_dict(estimated_dict, predicted)
    rf_l_table = pd.DataFrame(
        index=range(T_START, T_END - 1),
        columns=[period.name for period in Period],
        dtype=int,
    )
    for t in range(T_START, T_END - 1):
        training_panel = fp[fp["Year"] < t].copy()
        validation_panel = fp[fp["Year"] == t].copy()
        testing_panel = fp[fp["Year"] == t + 1].copy()

        for period in Period:
            response = f"RV_res^{period.name}"
            training_p_v = validate_panel(training_panel, response)
            validation_p_v = validate_panel(validation_panel, response)
            testing_p_v = validate_panel(testing_panel, response)

            (l, rf) = rf_grid_search(
                training_p_v,
                validation_p_v,
                period,
                random_state=random.randint(0, 6553400),
            )
            rf_l_table.loc[t, period.name] = l
            print(f"RF: l for year {t} series {period.name} is {l}")
            estimated = estimated_dict[period.name]
            estimated.loc[estimated["Year"] == t + 1, predicted] = rf.predict(
                testing_p_v[FEATURE_SET_ALL]
            )
    rf_l_table.to_pickle("../rf_l_table_stash.pkl")


def estimate_GBRT(fp: pd.DataFrame, estimated_dict: Dict[str, pd.DataFrame]):
    random.seed(20211003)
    dt = tree.DecisionTreeRegressor(
        max_features="log2", random_state=random.randint(0, 6553600)
    )
    lmbda = 0.001
    predicted = "RV_GB"
    expand_estimated_dict(estimated_dict, predicted)
    gb_l_table = pd.DataFrame(
        index=range(T_START, T_END - 1),
        columns=[period.name for period in Period],
        dtype=int,
    )
    gb_n_table = pd.DataFrame(
        index=range(T_START, T_END - 1),
        columns=[period.name for period in Period],
        dtype=int,
    )
    for t in range(T_START, T_END - 1):
        training_panel = fp[fp["Year"] < t].copy()
        validation_panel = fp[fp["Year"] == t].copy()
        testing_panel = fp[fp["Year"] == t + 1].copy()

        for period in Period:
            response = f"RV_res^{period.name}"
            training_p_v = validate_panel(training_panel, response)
            validation_p_v = validate_panel(validation_panel, response)
            testing_p_v = validate_panel(testing_panel, response)

            (l, n) = gb_grid_search(
                training_p_v,
                validation_p_v,
                period,
                random_state=random.randint(0, 6553400),
            )
            gb_l_table.loc[t, period.name] = l
            gb_n_table.loc[t, period.name] = n
            dt.max_depth = l
            print(f"GBRT: l for year {t} series {period.name} is {l}, len is {n}")
            training_p_v["residual"] = training_p_v[response]
            testing_p_v["predicted"] = 0
            for i in range(1, n + 1):
                training_panel_sample = training_p_v.sample(frac=0.5, replace=True)
                dt.fit(
                    training_panel_sample[FEATURE_SET_ALL],
                    training_panel_sample["residual"],
                )
                training_p_v["residual"] -= (
                    dt.predict(training_p_v[FEATURE_SET_ALL]) * lmbda
                )
                testing_p_v["predicted"] += (
                    dt.predict(testing_p_v[FEATURE_SET_ALL]) * lmbda
                )
            estimated = estimated_dict[period.name]
            estimated.loc[estimated["Year"] == t + 1, predicted] = testing_p_v[
                "predicted"
            ]
    gb_l_table.to_pickle("../gb_l_table_stash.pkl")
    gb_n_table.to_pickle("../gb_n_table_stash.pkl")


def main():
    fp: pd.DataFrame = pd.read_pickle("../feature_panel.pkl")
    fp["Year"] = (fp["Date"] / 10000).astype(int)
    # estimated_dict: Dict[str, pd.DataFrame] = {
    #     period.name: fp[["Stock", "Date", "Year", f"RV_res^{period.name}"]]
    #     .dropna()
    #     .rename(columns={f"RV_res^{period.name}": "RV_res"})
    #     for period in Period
    # }

    # for key in feature_set_dict.keys():
    #     print(f"Fitting {key}")
    #     estimate_OLS_Based(fp, estimated_dict, key)

    # for period in Period:
    #     estimate_MIDAS(fp, estimated_dict, period)

    # estimate_LASSO(fp, estimated_dict)

    # estimate_PCR(fp, estimated_dict)

    # estimate_RF(fp, estimated_dict)

    estimated_dict = pickle.load(open("../e_d.pkl", "rb"))
    estimate_GBRT(fp, estimated_dict)
    pickle.dump(estimated_dict, open("../e_d_1.pkl", "wb"))


if __name__ == "__main__":
    main()
