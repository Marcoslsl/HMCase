import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_period_of_day(elemen: datetime.datetime):
    element = elemen.hour
    if element > 0 and element <= 12:
        return 0
    elif element > 12 and element <= 18:
        return 1
    else:
        return 2


def prepare_data_to_regression(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data."""
    df = data.copy()

    # purchase_value
    df["purchase_value"] = df["purchase_value"].apply(
        lambda x: x * (-1) if x < 0 else x
    )

    # affiliate_commission_percentual
    df = df[~df["affiliate_commission_percentual"].isna()]

    # is_origin_page_social_network
    df["is_origin_page_social_network"] = (
        df["is_origin_page_social_network"]
        .apply(lambda x: x.split(",")[0])
        .astype("int")
        .copy()
    )

    # purchase_date
    df["period_of_day"] = df["purchase_date"].apply(get_period_of_day)

    # product_creation_date
    df["year"] = pd.to_datetime(df["product_creation_date"]).dt.year.astype(
        "int64"
    )
    df["month"] = pd.to_datetime(df["product_creation_date"]).dt.month.astype(
        "int64"
    )
    df["day"] = pd.to_datetime(df["product_creation_date"]).dt.day.astype(
        "int64"
    )

    # drop columns
    df = df.drop(
        ["purchase_id", "Venda", "product_creation_date", "purchase_date"],
        axis=1,
    )

    return df


def mean_absolute_percentage_erro(y, yhat):
    """MAPE"""
    return np.mean(np.abs((y - yhat) / y))


def ml_error(model_name, y, yhat):
    """Get MAE, MAPE and RMSE."""
    mae = mean_absolute_error(y, yhat)
    mape = mean_absolute_percentage_erro(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))

    return pd.DataFrame(
        {"Model Name": model_name, "MAE": mae, "MAPE": mape, "RMSE": rmse},
        index=[0],
    )
