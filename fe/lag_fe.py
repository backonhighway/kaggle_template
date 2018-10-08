from rstudio.common import pred_cols
import numpy as np
import pandas as pd

class LagFe:

    def do_fe(self, train, test):
        train = self.do_fe_for_train(train)
        test = self.do_fe_for_test(train, test)
        return train, test

    @staticmethod
    def do_fe_for_train(df):
        for col in pred_cols.LAG_COLS:
            tar_col_name = "lag_tar_" + col
            hit_col_name = "lag_hits_" + col
            df["temp_shifted"] = df.groupby(col)["target"].shift(1)
            df[tar_col_name] = df[col].rolling(400, min_periods=1).mean()
            df["temp_shifted"] = df.groupby(col)["target2"].shift(1)
            df[hit_col_name] = df[col].rolling(400, min_periods=1).mean()

        df.drop("temp_shifted", axis=1, inplace=True)
        return df

    @staticmethod
    def do_fe_for_test(train, test):
        for col in pred_cols.LAG_COLS:
            tar_col_name = "lag_tar_" + col
            hit_col_name = "lag_hits_" + col
            test["temp_shifted"] = test.groupby(col)["target2"].shift(1)
            test[hit_col_name] = test[col].rolling(400, min_periods=1).mean()

            train_target = train.groupby(col)["target"].mean().reset_index()
            rename_dict = {"target": tar_col_name}
            train_target.rename(columns=rename_dict, inplace=True)
            test = pd.merge(test, train_target, on=col, how="left")

        train.drop("target", axis=1, inplace=True)
        test.drop("target2", axis=1, inplace=True)
        return test
