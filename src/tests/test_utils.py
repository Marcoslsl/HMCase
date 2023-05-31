from src.utils import numeric_statistics
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def data():
    num_linhas = 10
    num_colunas = 5

    dados_aleatorios = np.random.rand(num_linhas, num_colunas)
    df = pd.DataFrame(dados_aleatorios)

    return df


def test_numeric_statistics(data):
    columns = [
        "type",
        "Unique_Values",
        "Mean",
        "Median",
        "Std",
        "Min",
        "Max",
        "Range",
        "Skew",
        "Kurtosis",
    ]
    df = numeric_statistics(data)
    assert columns == df.columns.tolist()
