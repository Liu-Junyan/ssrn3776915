import sklearn
from constants import FEATURE_SET_ALL
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from constants import Period
from sklearn import linear_model
from sklearn import decomposition


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
    log_space = np.logspace(-5, 2, 200)
    log_space = np.insert(log_space, 0, 0)
    lmbda_space = log_space[log_space < 0.4]
    lmbda_R2_profile = pd.Series(index=lmbda_space, dtype=float)
    for lmbda in lmbda_space:
        if lmbda != 0:
            lasso.alpha = lmbda
            lasso.fit(
                training_panel[FEATURE_SET_ALL], training_panel[f"RV_res^{period.name}"]
            )
            predicted_array: np.ndarray = lasso.predict(
                validation_panel[FEATURE_SET_ALL]
            )
        else:
            lm.fit(
                training_panel[FEATURE_SET_ALL], training_panel[f"RV_res^{period.name}"]
            )
            predicted_array: np.ndarray = lm.predict(validation_panel[FEATURE_SET_ALL])
        lmbda_R2_profile.loc[lmbda] = R_squared_OOS(
            estimated["RV_res"], estimated["RV_HAR"], predicted_array
        )
    return lmbda_R2_profile.idxmax()


def pca_grid_search(
    training_panel: pd.DataFrame,
    validation_panel: pd.DataFrame,
    estimated: pd.DataFrame,
    period: Period,
) -> int:
    pca = decomposition.PCA()
    lm = linear_model.LinearRegression()
    n_space = list(range(1, len(FEATURE_SET_ALL)))
    n_MSE_profile = pd.Series(index=n_space, dtype=float)
    for n in n_space:
        pca.n_components = n
        pca.fit(training_panel[FEATURE_SET_ALL])
        training_X = pca.transform(training_panel[FEATURE_SET_ALL])
        validation_X = pca.transform(validation_panel[FEATURE_SET_ALL])
        lm.fit(training_X, training_panel[f"RV_res^{period.name}"])
        predicted_array = lm.predict(validation_X)
        n_MSE_profile.loc[n] = MSE(estimated["RV_res"], predicted_array)
    return n_MSE_profile.idxmin()


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
