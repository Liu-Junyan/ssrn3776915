from constants import FEATURE_SET
from typing import Dict
import numpy as np
import pandas as pd
from constants import Period
from sklearn import linear_model


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

    for period in Period:
        estimated = estimated_dict[period.name]
        try:
            estimated.insert(
                len(estimated.columns), f"{predicted}^{period.name}", np.nan
            )
        except ValueError:  # Already exists
            return


def lasso_grid_search(
    training_panel: pd.DataFrame,
    validation_panel: pd.DataFrame,
    estimated: pd.DataFrame,
    period: str,
) -> np.float64:
    lasso = linear_model.Lasso()
    param_space = np.logspace(-5, 2, 200)
    lmbda_profile = pd.Series(index=param_space, dtype=float)
    for param in param_space:
        lasso.alpha = param
        lasso.fit(training_panel[FEATURE_SET], training_panel[f"RV_res^{period}"])
        predicted: np.ndarray = lasso.predict(validation_panel[FEATURE_SET])
        lmbda_profile.loc[param] = R_squared_OOS(
            estimated[f"RV_res^{period}"], estimated[f"RV_HAR^{period}"], predicted
        )
    return lmbda_profile.idxmax()


def R_squared_OOS(res, benchmark, predicted) -> float:
    return 1 - np.sum(np.square(res - predicted)) / np.sum(np.square(res - benchmark))
