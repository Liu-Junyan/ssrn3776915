import copy
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn import ensemble
from sklearn import linear_model
from sklearn import tree

from constants import FEATURE_SET_ALL
from constants import Period


def validate_panel(panel: pd.DataFrame, response: str) -> pd.DataFrame:
    """Validating a panel by dropping rows of it where the response variables don't exist.

    Args:
        panel (pd.DataFrame): A panel of features and response variables.
        response (str): The name of response variable

    Returns:
        pd.DataFrame: Validated panel.
    """
    return panel.dropna(subset=[response])


def standardize(panel: pd.DataFrame, mean: float, std: float) -> pd.DataFrame:
    """Standardize a panel of features.

    Args:
        panel (pd.DataFrame): A panel of features.
        mean (float): Mean, usually the mean of the training set.
        std (float): Standard deviation, usually the standard deviation of the training set.

    Returns:
        pd.DataFrame: Standardized panel.
    """
    return (panel - mean) / std


def expand_estimated_dict(estimated_dict: Dict[str, pd.DataFrame], predicted: str):
    """Expand the estimated dict by adding new predicted variable to each estimated panel.

    Args:
        estimated_dict (Dict[str, pd.DataFrame]): A dictionary of predicted results.
        predicted (str): The name of predicted variable.
    """
    for period in Period:
        estimated = estimated_dict[period.name]
        try:
            estimated.insert(len(estimated.columns), predicted, np.nan)
        except ValueError:  # Already exists
            return


def lasso_grid_search(
    training_panel: pd.DataFrame,
    validation_panel: pd.DataFrame,
    estimated: pd.DataFrame,
    period: Period,
) -> np.float64:
    """Use grid search to find optimal lambda for LASSO.

    Args:
        training_panel (pd.DataFrame): Validated training panel.
        validation_panel (pd.DataFrame): Validated validation panel.
        estimated (pd.DataFrame): Estimated panel.
        period (Period): Forecast horizon.

    Returns:
        np.float64: Optimal lambda.
    """
    lasso = linear_model.Lasso()
    lm = linear_model.LinearRegression()
    response = f"RV_res^{period.name}"
    log_space = np.logspace(-5, 2, 200)
    log_space = np.insert(log_space, 0, 0)
    lmbda_space = log_space[log_space < 0.4]
    lmbda_R2_profile = pd.Series(index=lmbda_space, dtype=float)
    for lmbda in lmbda_space:
        if lmbda != 0:
            lasso.alpha = lmbda
            lasso.fit(training_panel[FEATURE_SET_ALL], training_panel[response])
            predicted_array: np.ndarray = lasso.predict(
                validation_panel[FEATURE_SET_ALL]
            )
        else:
            lm.fit(training_panel[FEATURE_SET_ALL], training_panel[response])
            predicted_array: np.ndarray = lm.predict(validation_panel[FEATURE_SET_ALL])
        lmbda_R2_profile.loc[lmbda] = R_squared_OOS(
            estimated["RV_res"], estimated["RV_HAR"], predicted_array
        )
    return lmbda_R2_profile.idxmax()


def pca_grid_search(
    training_panel: pd.DataFrame, validation_panel: pd.DataFrame, period: Period
) -> int:
    pca = decomposition.PCA()
    lm = linear_model.LinearRegression()
    response = f"RV_res^{period.name}"
    n_space = list(range(1, len(FEATURE_SET_ALL)))
    n_MSE_profile = pd.Series(index=n_space, dtype=float)
    for n in n_space:
        pca.n_components = n
        pca.fit(training_panel[FEATURE_SET_ALL])
        training_X = pca.transform(training_panel[FEATURE_SET_ALL])
        validation_X = pca.transform(validation_panel[FEATURE_SET_ALL])
        lm.fit(training_X, training_panel[response])
        predicted_array = lm.predict(validation_X)
        n_MSE_profile.loc[n] = MSE(validation_panel[response], predicted_array)
    return n_MSE_profile.idxmin()


def rf_grid_search(
    training_panel: pd.DataFrame,
    validation_panel: pd.DataFrame,
    period: Period,
    random_state: int,
) -> Tuple[int, ensemble.RandomForestRegressor]:
    rf = ensemble.RandomForestRegressor(
        n_estimators=100,
        max_features="log2",
        bootstrap=True,
        max_samples=0.5,
        random_state=random_state,
        n_jobs=-1,
    )
    response = f"RV_res^{period.name}"
    l_space = list(range(1, 21))
    l_MSE_profile = pd.Series(index=l_space, dtype=float)
    models_dict = {}
    for l in l_space:
        rf.max_depth = l
        rf.fit(training_panel[FEATURE_SET_ALL], training_panel[response])
        predicted_array: np.ndarray = rf.predict(validation_panel[FEATURE_SET_ALL])
        l_MSE_profile.loc[l] = MSE(validation_panel[response], predicted_array)
        models_dict[l] = copy.deepcopy(rf)
    l = l_MSE_profile.idxmin()
    return (l, models_dict[l])


def rf_profile(
    training_panel: pd.DataFrame, validation_panel: pd.DataFrame, period: Period
):
    rf = ensemble.RandomForestRegressor(max_depth=20, max_features="log2")
    response = f"RV_res^{period.name}"
    n_space = list(range(100, 201, 5))
    n_MSE_profile = pd.Series(index=n_space, dtype=float)
    for n in n_space:
        print(n)
        rf.n_estimators = n
        rf.fit(training_panel[FEATURE_SET_ALL], training_panel[response])
        predicted_array: np.ndarray = rf.predict(validation_panel[FEATURE_SET_ALL])
        n_MSE_profile.loc[n] = MSE(validation_panel[response], predicted_array)
    return n_MSE_profile


def gb_grid_search(
    training_panel: pd.DataFrame,
    validation_panel: pd.DataFrame,
    period: Period,
    random_state: int,
) -> Tuple[int, int]:
    dt = tree.DecisionTreeRegressor(max_features="log2", random_state=random_state)
    response = f"RV_res^{period.name}"
    lmbda = 0.001
    l_space = list(range(1, 6))
    l_MSE_profile = pd.Series(index=l_space, dtype=float)
    len_dict: Dict[int, int] = {}
    for l in l_space:
        dt.max_depth = l
        training_panel["residuals"] = training_panel[response]
        validation_panel["predicted"] = 0
        MSE_profile = np.ndarray(shape=(20000,))
        max_len = 0
        for i in range(1, 20001):
            max_len = i
            training_panel_sample = training_panel.sample(frac=0.5, replace=True)
            dt.fit(
                training_panel_sample[FEATURE_SET_ALL],
                training_panel_sample["residuals"],
            )
            training_panel["residuals"] -= (
                dt.predict(training_panel[FEATURE_SET_ALL]) * lmbda
            )
            validation_panel["predicted"] += (
                dt.predict(validation_panel[FEATURE_SET_ALL]) * lmbda
            )
            MSE_profile[i - 1] = MSE(
                validation_panel[response], validation_panel["predicted"]
            )
            if should_early_stop(MSE_profile, i - 1):
                break
        l_MSE_profile[l] = MSE(
            validation_panel[response], validation_panel["predicted"]
        )
        len_dict[l] = max_len
    l = l_MSE_profile.idxmin()
    return (l, len_dict[l])


def should_early_stop(MSE_profile: np.ndarray, i: int) -> bool:
    n = 50
    tol = 0.001
    if i < n:
        return False
    else:
        sub = MSE_profile[i - 50 : i]
        return np.max(sub - MSE_profile[i]) < tol


def R_squared_OOS(res, benchmark, predicted) -> float:
    """Calculate Out-of-sample R squared.

    Args:
        res (Array-like object): Response variables.
        benchmark (Array-like object): Benchmark variables.
        predicted (Array-like object): Predicted variables.

    Returns:
        float: Out-of-sample R squared.
    """
    return 1 - np.sum(np.square(res - predicted)) / np.sum(np.square(res - benchmark))


def MSE(res, predicted) -> float:
    return 1 / len(res) * np.sum(np.square(res - predicted))
