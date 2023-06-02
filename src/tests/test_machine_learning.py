from src.utils import (
    get_period_of_day,
    mean_absolute_percentage_erro,
    ml_error,
)
import pandas as pd
import numpy as np
import pytest


@pytest.mark.parametrize(
    "hour, expected",
    [
        [pd.Timestamp(year=2019, month=2, hour=10, day=1), 0],
        [pd.Timestamp(year=2019, month=2, hour=14, day=1), 1],
        [pd.Timestamp(year=2019, month=2, hour=19, day=1), 2],
    ],
)
def test_get_period_of_day(hour, expected):
    assert get_period_of_day(hour) == expected


def test_mean_absolute_percentage_erro():
    y = np.array([50, 75, 100])
    yhat = np.array([60, 80, 95])
    assert np.round(mean_absolute_percentage_erro(y, yhat), 2) == 0.11


def test_ml_error():
    # Dados de exemplo
    model_name = "Random Forest"
    y = np.array([10, 20, 30, 40, 50])
    yhat = np.array([12, 18, 25, 42, 48])

    # Valores esperados
    expected_mae = 2.6
    expected_mape = 0.11133333333333335
    expected_rmse = 2.8635642126552705

    # Executar a função
    result = ml_error(model_name, y, yhat)

    # Verificar se os resultados estão corretos
    assert result["Model Name"].iloc[0] == model_name
    assert result["MAE"].iloc[0] == expected_mae
    assert result["MAPE"].iloc[0] == expected_mape
    assert result["RMSE"].iloc[0] == expected_rmse
