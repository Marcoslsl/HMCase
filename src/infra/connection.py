import pandas as pd
from sqlalchemy.sql import text
from sqlalchemy import create_engine


class DBConnection:
    def __init__(self, user, password, host, port, schema) -> None:
        self.__user = user
        self.__password = password
        self.__host = host
        self.__port = port
        self.__schema = schema

        self.url_conexao = f"""mysql+pymysql://{self.__user}:{self.__password}@{self.__host}:{self.__port}/{self.__schema}"""
        self.__engine = create_engine(self.url_conexao, echo=True)

    def get_engine(self):
        """Get engine."""
        return self.__engine

    def get_data(self):
        """Get data from database."""

        sql = """
            SELECT * FROM sales_data;
        """

        with self.__engine.connect() as conn:
            query = conn.execute(text(sql))
        df = pd.DataFrame(query.fetchall())
        return df
