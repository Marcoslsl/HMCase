import datetime
import pandas as pd
import numpy as np


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data."""
    df = data.copy()

    # purchase_value
    df["purchase_value_positiva"] = df["purchase_value"].apply(
        lambda x: x * (-1) if x < 0 else x
    )

    # affiliate_commission_percentual
    df = df[~df["affiliate_commission_percentual"].isna()]

    # columns to str
    columns_to_str = [
        "purchase_id",
        "product_id",
        "affiliate_id",
        "producer_id",
        "buyer_id",
    ]
    df.loc[:, columns_to_str] = df.loc[:, columns_to_str].astype(str).copy()

    # is_origin_page_social_network
    df["is_origin_page_social_network"] = (
        df["is_origin_page_social_network"]
        .apply(lambda x: x.split(",")[0])
        .astype("int")
        .copy()
    )

    return df


class RFMModel:
    """RFMModel."""

    def __init__(
        self,
        id_colum: str = "buyer_id",
        monetary_column: str = "purchase_value_positiva",
        recency_colum: str = "recency",
        frequency_colum: str = "frequency",
        purchase_date_column: str = "purchase_date",
    ) -> None:
        """Construct."""
        self.id_colum = id_colum
        self.monetary_colum = monetary_column
        self.recency_colum = recency_colum
        self.frequency_colum = frequency_colum
        self.purchase_date_column = purchase_date_column

    def get_rfm_model(self, df_prep):
        """Main pipeline to get the RFMModel."""
        df_rfm = self.get_rfm_df(df_prep)
        df_rfm_final = self.get_scores_rfm(df_rfm)
        df_rfm_score_final = self.get_final_rfm_score(df_rfm_final)
        df_final = RFMModel.get_clusters(df_rfm_score_final)
        return df_final

    def get_rfm_df(self, df_prep: pd.DataFrame) -> pd.DataFrame:
        """Get the rfm dataframe.

        Parameters
        ----------
        df_prep: pd.DataFrame
            Pandas DataFrame.
        """
        df_cluster = df_prep[
            [self.id_colum, self.purchase_date_column, self.monetary_colum]
        ]
        df_cluster.loc[:, self.purchase_date_column] = pd.to_datetime(
            df_cluster[self.purchase_date_column]
        ).copy()

        recency = (
            datetime.datetime.now() - df_cluster[self.purchase_date_column]
        )
        recency = recency.apply(lambda x: x.days)
        df_cluster.loc[:, self.recency_colum] = recency.copy()

        df_frequency = df_cluster[self.id_colum].value_counts().reset_index()
        df_recency = (
            df_cluster[[self.id_colum, self.recency_colum]]
            .groupby(self.id_colum)
            .min()
            .reset_index()
        )
        df_monetary = (
            df_cluster[[self.id_colum, self.monetary_colum]]
            .groupby(self.id_colum)
            .sum()
            .reset_index()
        )

        df_rf = df_frequency.merge(df_recency, how="inner", on=self.id_colum)
        df_rfm = df_rf.merge(df_monetary, how="inner", on=self.id_colum)
        df_rfm.rename(
            columns={"count": "frequency", self.monetary_colum: "monetary"},
            inplace=True,
        )

        self.monetary_colum = "monetary"

        return df_rfm

    def get_scores_rfm(
        self, df_rfm: pd.DataFrame, num_parts: int = 5
    ) -> pd.DataFrame:
        """Get the scores from rfm data frame.

        Parameters
        ----------
        id_colum: str, default = 'buyer_id'
        num_parts: int, default = 5
        """
        df_freq = df_rfm[[self.id_colum, self.frequency_colum]].sort_values(
            self.frequency_colum
        )
        df_mone = df_rfm[[self.id_colum, self.monetary_colum]].sort_values(
            self.monetary_colum
        )
        df_rece = df_rfm[[self.id_colum, self.recency_colum]].sort_values(
            self.recency_colum, ascending=False
        )

        parts_freq = np.array_split(df_freq, num_parts, axis=0)
        parts_mone = np.array_split(df_mone, num_parts, axis=0)
        parts_rece = np.array_split(df_rece, num_parts, axis=0)

        df_scores_freq = pd.DataFrame()
        df_scores_mone = pd.DataFrame()
        df_scores_rece = pd.DataFrame()

        parts_list = [parts_freq, parts_mone, parts_rece]
        dfs_list = [df_scores_freq, df_scores_mone, df_scores_rece]
        names_scores_list = [
            "frequency_score",
            "monetary_score",
            "recency_score",
        ]

        df_final = pd.DataFrame()
        for parts, df, score_name in zip(
            parts_list, dfs_list, names_scores_list
        ):
            score = 1
            for part in parts:
                part[score_name] = score
                score += 1
                df = pd.concat([df, part])
            if score_name == "frequency_score":
                df_final = df
            else:
                df_final = df_final.merge(df, how="inner", on="buyer_id")

        return df_final[[self.id_colum] + names_scores_list]

    def get_final_rfm_score(
        self,
        df_rfm_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get the final rfm score.

        Parameters
        ----------
        df_rfm_scores: pd.DataFrame
        id_colum: str, default = 'buyer_id'
        """
        df_rfm_scores["RFM_score"] = (
            df_rfm_scores.drop(self.id_colum, axis=1).sum(axis=1) / 3
        )
        return df_rfm_scores

    @staticmethod
    def get_names_of_cluster(element: float) -> str:
        """Get the names of clusters.

        Parameters
        ----------
        elemnt: float
            The between F and M scores.
        """
        if element <= 2:
            return "Hibernando"
        elif element > 2 and element <= 4:
            return "Promissor"
        else:
            return "Campeao"

    @staticmethod
    def get_clusters(df: pd.DataFrame) -> pd.DataFrame:
        """Get the clusters."""
        df_aux = df.copy()
        df_aux["FM_mean"] = (
            df_aux[["monetary_score", "frequency_score"]].sum(axis=1) / 2
        )
        df_aux["clusters"] = df_aux["FM_mean"].apply(
            RFMModel.get_names_of_cluster
        )
        return df_aux
