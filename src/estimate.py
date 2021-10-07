"""
Usage: cd src && python3 estimate.py
"""
import pickle
import random
from multiprocessing import Process, Manager
from typing import Dict, List

import pandas as pd

from constants import T_START, T_END
from estimate_util import *

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
    lasso_n_table = pd.DataFrame(
        index=range(T_START, T_END - 1),
        columns=[period.name for period in Period],
        dtype=int,
    )
    if use_stash:
        lmbda_table_stash: pd.DataFrame = pickle.load(
            open("../stash/lmbda_table_stash.pkl", "rb")
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
                model = lasso
            else:  # Collapse to OLS
                lm.fit(training_p_v[FEATURE_SET_ALL], training_p_v[response])
                estimated.loc[estimated["Year"] == t + 1, predicted] = lm.predict(
                    testing_p_v[FEATURE_SET_ALL]
                )
                model = lm
            coef_list = model.coef_
            lasso_n_table.loc[t, period.name] = len(coef_list[coef_list != 0])
    lasso_n_table.to_csv("../csv/lasso_n.csv")
    lmbda_table.to_pickle("../stash/lmbda_table_stash.pkl")


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
            open("../stash/pca_n_table_stash.pkl", "rb")
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
    pca_n_table.to_pickle("../stash/pca_n_table_stash.pkl")


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
    rf_l_table.to_pickle("../stash/rf_l_table_stash.pkl")


def estimate_GBRT(fp: pd.DataFrame, estimated_dict: Dict[str, pd.DataFrame]):
    random.seed(20211003)
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
        print(f"Year: {t}")
        training_panel = fp[fp["Year"] < t].copy()
        validation_panel = fp[fp["Year"] == t].copy()
        testing_panel = fp[fp["Year"] == t + 1].copy()

        manager = Manager()
        return_dict = manager.dict()

        P_list = [
            Process(
                target=GBRT_helper,
                args=(
                    training_panel,
                    validation_panel,
                    testing_panel,
                    period,
                    return_dict,
                ),
            )
            for period in Period
        ]

        for P in P_list:
            P.start()

        for P in P_list:
            P.join()

        for period, value in return_dict.items():
            estimated = estimated_dict[period.name]
            gb_l_table.loc[t, period.name] = value[0]
            gb_n_table.loc[t, period.name] = value[1]
            estimated.loc[estimated["Year"] == t + 1, predicted] = value[2]

    gb_l_table.to_pickle("../stash/gb_l_table_stash.pkl")
    gb_n_table.to_pickle("../stash/gb_n_table_stash.pkl")


def GBRT_helper(
    training_panel: pd.DataFrame,
    validation_panel: pd.DataFrame,
    testing_panel: pd.DataFrame,
    period: Period,
    return_dict,
):
    lmbda = 0.001
    response = f"RV_res^{period.name}"
    training_p_v = validate_panel(training_panel, response)
    validation_p_v = validate_panel(validation_panel, response)
    testing_p_v = validate_panel(testing_panel, response)
    (l, n, dt_list) = gb_grid_search(
        training_p_v, validation_p_v, period, random_state=random.randint(0, 6553400),
    )
    print(f"GBRT: l for series {period.name} is {l}, len is {n}")
    predicted_array = np.zeros(testing_p_v.shape[0])
    for dt in dt_list:
        predicted_array += dt.predict(testing_p_v[FEATURE_SET_ALL]) * lmbda
    return_dict[period] = (l, n, predicted_array)


def main():
    fp: pd.DataFrame = pd.read_pickle("../stash/feature_panel.pkl")
    fp["Year"] = (fp["Date"] / 10000).astype(int)
    estimated_dict: Dict[str, pd.DataFrame] = {
        period.name: fp[["Stock", "Date", "Year", f"RV_res^{period.name}"]]
        .dropna()
        .rename(columns={f"RV_res^{period.name}": "RV_res"})
        for period in Period
    }

    for key in feature_set_dict.keys():
        print(f"Fitting {key}")
        estimate_OLS_Based(fp, estimated_dict, key)

    for period in Period:
        estimate_MIDAS(fp, estimated_dict, period)

    estimate_LASSO(fp, estimated_dict)

    estimate_PCR(fp, estimated_dict)

    estimate_RF(fp, estimated_dict)

    estimate_GBRT(fp, estimated_dict)
    pickle.dump(estimated_dict, open("../stash/e_d.pkl", "wb"))


if __name__ == "__main__":
    main()
