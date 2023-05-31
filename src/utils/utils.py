"""Utils."""
import pandas as pd


def numeric_statistics(df: pd.DataFrame = None) -> pd.DataFrame:
    """Make a descriptive analysis on numeric data.

    Parameters
    ----------
    df: pd.DataFrame, default=None.
        Pandas dataframe.
    """
    dic = {
        "type": df.dtypes.values,
        "Unique_Values": df.nunique().values,
        "Mean": df.mean(),
        "Median": df.median(),
        "Std": df.std(),
        "Min": df.min(),
        "Max": df.max(),
        "Range": df.max() - df.min(),
        "Skew": df.skew(),
        "Kurtosis": df.kurtosis(),
    }

    return pd.DataFrame(dic, index=df.columns)
