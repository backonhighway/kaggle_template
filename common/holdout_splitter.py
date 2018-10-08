import pandas as pd


class GoldenSplitter:

    @staticmethod
    def get_holdout_split(df:pd.DataFrame, from_date, to_date=20180101):
        train = df[df["date"] < from_date]
        holdout = df[(df["date"] >= from_date) & (df["date"] < to_date)]
        return train, holdout
