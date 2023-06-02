from src.utils import numeric_statistics, RFMModel
import pandas as pd
import numpy as np
import pytest
import json


@pytest.mark.parametrize(
    "element, expected", [[1, "Hibernando"], [3, "Promissor"], [5, "Campeao"]]
)
def test_get_names_of_cluster(element, expected):
    response = RFMModel.get_names_of_cluster(element)
    assert response == expected


def test_get_clusters():
    json_teste = {
        "buyer_id": {
            0: "4984688",
            1: "5917688",
            2: "718553",
            3: "5917691",
            4: "5550404",
        },
        "frequency_score": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        "monetary_score": {0: 4, 1: 1, 2: 2, 3: 5, 4: 3},
        "recency_score": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        "RFM_score": {
            0: 2.0,
            1: 1.6666666666666667,
            2: 2.6666666666666665,
            3: 4.333333333333333,
            4: 4.333333333333333,
        },
    }

    df = pd.DataFrame(json_teste)
    rfm_model = RFMModel(monetary_column="monetary")
    df_rfm = rfm_model.get_clusters(df)
    assert "clusters" in df_rfm.columns.tolist()


@pytest.fixture
def data():
    data = {
        "purchase_id": {
            "199": "10839481",
            "200": "10839484",
            "201": "10839485",
        },
        "product_id": {"199": "118733", "200": "181119", "201": "155827"},
        "affiliate_id": {"199": "958412", "200": "213339", "201": "2035132"},
        "producer_id": {"199": "958412", "200": "213339", "201": "2035132"},
        "buyer_id": {"199": "4984688", "200": "5917688", "201": "718553"},
        "purchase_date": {
            "199": "2016-01-01 00:04:44",
            "200": "2016-01-01 00:05:27",
            "201": "2016-01-01 00:05:29",
        },
        "product_creation_date": {
            "199": "2014-10-04 13:43:24",
            "200": "2015-10-22 11:05:58",
            "201": "2015-06-08 14:28:20",
        },
        "product_category": {
            "199": "Phisical book",
            "200": "Phisical book",
            "201": "Podcast",
        },
        "product_niche": {
            "199": "Organization",
            "200": "Organization",
            "201": "Anxiety management",
        },
        "purchase_value": {"199": -0.4288, "200": 0.204131, "201": -0.37793},
        "affiliate_commission_percentual": {
            "199": 0.0,
            "200": 0.0,
            "201": 0.0,
        },
        "purchase_device": {
            "199": "eReaders",
            "200": "Desktop",
            "201": "eReaders",
        },
        "purchase_origin": {
            "199": "Origin adf0",
            "200": "Origin 3c5a",
            "201": "Origin 14ad",
        },
        "is_origin_page_social_network": {"199": 0, "200": 0, "201": 0},
        "Venda": {"199": 1, "200": 1, "201": 1},
        "purchase_value_positiva": {
            "199": 0.4288,
            "200": 0.204131,
            "201": 0.37793,
        },
    }
    return data


def test_get_rfm_df(data):
    json_teste = {
        "buyer_id": {"0": "4984688", "1": "5917688", "2": "718553"},
        "frequency": {"0": 1, "1": 1, "2": 1},
        "recency": {"0": 2709, "1": 2709, "2": 2709},
        "monetary": {"0": 0.4288, "1": 0.204131, "2": 0.37793},
    }
    df = pd.DataFrame(data)
    rfm_model = RFMModel()
    df_rfm = rfm_model.get_rfm_df(df)
    df_rfm_json = json.loads(df_rfm.to_json())
    assert json_teste == df_rfm_json


def test_get_scores_rfm():
    json_teste = {
        "buyer_id": {
            0: "4984688",
            1: "5917688",
            2: "718553",
            3: "5917691",
            4: "5550404",
        },
        "frequency": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        "recency": {0: 2709, 1: 2709, 2: 2709, 3: 2709, 4: 2709},
        "monetary": {
            0: 0.4288,
            1: 0.204131,
            2: 0.37793,
            3: 0.503885,
            4: 0.415472,
        },
    }
    json_response = {
        "buyer_id": {
            0: "4984688",
            1: "5917688",
            2: "718553",
            3: "5917691",
            4: "5550404",
        },
        "frequency_score": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        "monetary_score": {0: 4, 1: 1, 2: 2, 3: 5, 4: 3},
        "recency_score": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
    }
    df = pd.DataFrame(json_teste)
    rfm_model = RFMModel(monetary_column="monetary")
    df_rfm = rfm_model.get_scores_rfm(df)
    df_rfm_scores_json = df_rfm.to_dict()
    assert df_rfm_scores_json == json_response


def test_get_final_rfm_score():
    json_teste = {
        "buyer_id": {
            0: "4984688",
            1: "5917688",
            2: "718553",
            3: "5917691",
            4: "5550404",
        },
        "frequency_score": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        "monetary_score": {0: 4, 1: 1, 2: 2, 3: 5, 4: 3},
        "recency_score": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
    }
    json_response = {
        "buyer_id": {
            0: "4984688",
            1: "5917688",
            2: "718553",
            3: "5917691",
            4: "5550404",
        },
        "frequency_score": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        "monetary_score": {0: 4, 1: 1, 2: 2, 3: 5, 4: 3},
        "recency_score": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        "RFM_score": {
            0: 2.0,
            1: 1.6666666666666667,
            2: 2.6666666666666665,
            3: 4.333333333333333,
            4: 4.333333333333333,
        },
    }
    df = pd.DataFrame(json_teste)
    rfm_model = RFMModel(monetary_column="monetary")
    df_rfm = rfm_model.get_final_rfm_score(df)
    df_rfm_scores_json = df_rfm.to_dict()
    assert df_rfm_scores_json == json_response
