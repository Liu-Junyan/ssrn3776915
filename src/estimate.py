"""[summary]
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn import linear_model
from constants import Period, FEATURE_SET, T_START, T_END
from estimate_util import *
import pickle

feature_set_dict: Dict[str, List[str]] = {
    "HAR": ["RV^d", "RV^w", "RV^m", "RV^q"],
    "SHAR": ["RVP^d", "RVN^d", "RV^w", "RV^m", "RV^q"],
    "HARQ": ["RV^d", "RV^w", "RV^m", "RV^q", "HARQ^d", "HARQ^w", "HARQ^m", "HARQ^q"],
    "HExpGl": ["ExpRV^1", "ExpRV^5", "ExpRV^25", "ExpRV^125", "ExpGlRV"],
    "All": FEATURE_SET,
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


def estimate_HAR(fp: pd.DataFrame, estimated_dict: Dict[str, pd.DataFrame]):
    """Estimate by HAR.

    Args:
        fp (pd.DataFrame): A panel of features and response variables.
        estimated_dict (Dict[str, pd.DataFrame]): A dict of estimated results. MUTATED.
    """
    lm = linear_model.LinearRegression()
    predicted = "RV_HAR"
    expand_estimated_dict(estimated_dict, predicted)
    for period in Period:
        response = f"RV_res^{period.name}"
        estimated = estimated_dict[period.name]
        # t for testing set
        for t in range(T_START, T_END):
            training_panel = fp[fp["Year"] < t]
            testing_panel = fp[fp["Year"] == t]
            training_p_v = validate_panel(training_panel, response)
            testing_p_v = validate_panel(testing_panel, response)
            lm.fit(training_p_v[FEATURE_SET], training_p_v[response])
            estimated.loc[estimated["Year"] == t, predicted] = lm.predict(
                testing_p_v[FEATURE_SET]
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
    lmbda_table_stash = pickle.load(open("../lmbda_table_stash.pkl", "rb"))
    # Under training-validation-testing scheme, t = 2008...2015. (testing: 2007...t-1; validation: t; testing: t+1)
    # t for validation set
    for t in range(T_START, T_END - 1):
        training_panel = fp[fp["Year"] < t].copy()
        validation_panel = fp[fp["Year"] == t].copy()
        testing_panel = fp[fp["Year"] == t + 1].copy()

        training_mean = training_panel[FEATURE_SET].mean()
        training_std = training_panel[FEATURE_SET].std()

        training_panel[FEATURE_SET] = standardize(
            training_panel[FEATURE_SET], training_mean, training_std
        )
        validation_panel[FEATURE_SET] = standardize(
            validation_panel[FEATURE_SET], training_mean, training_std
        )
        testing_panel[FEATURE_SET] = standardize(
            testing_panel[FEATURE_SET], training_mean, training_std
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
                else (lmbda_table_stash.loc[t, period.name])
            )
            lmbda_table.loc[t, period.name] = lmbda
            print(f"Lambda for year {t} series {period.name} is {lmbda}")
            estimated = estimated_dict[period.name]
            if lmbda != 0:
                lasso.alpha = lmbda
                lasso.fit(training_p_v[FEATURE_SET], training_p_v[response])
                estimated.loc[estimated["Year"] == t + 1, predicted] = lasso.predict(
                    testing_p_v[FEATURE_SET]
                )
            else:  # Collapse to OLS
                lm.fit(training_p_v[FEATURE_SET], training_p_v[response])
                estimated.loc[estimated["Year"] == t + 1, predicted] = lm.predict(
                    testing_p_v[FEATURE_SET]
                )
    lmbda_table.to_pickle("../lmbda_table.pkl")


def main():
    fp: pd.DataFrame = pd.read_pickle("../feature_panel.pkl")
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

    pickle.dump(estimated_dict, open("../e_d.pkl", "wb"))

    # estimated_dict = pickle.load(open("../e_d.pkl", "rb"))

    # estimate_LASSO(fp, estimated_dict, use_stash=True)
    # pickle.dump(estimated_dict, open("../e_d_1.pkl", "wb"))


if __name__ == "__main__":
    main()
