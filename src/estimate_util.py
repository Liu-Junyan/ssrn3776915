import pandas as pd


def validate_panel(panel: pd.DataFrame, response: str) -> pd.DataFrame:
    """Validating panel by dropping rows of it where the response variables don't exist.

    Args:
        panel (pd.DataFrame): A panel of features and response variables.
        response (str): The name of response variable

    Returns:
        pd.DataFrame: Validated panel.
    """
    return panel.dropna(subset=[response])
