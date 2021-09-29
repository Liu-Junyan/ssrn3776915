import pandas as pd

def validate_panel(panel: pd.DataFrame, response: str) -> pd.DataFrame:
    return panel.dropna(subset=[response])