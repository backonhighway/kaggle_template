from rstudio.common import pred_cols


class UserFe:

    def do_fe(self, train, test):
        train = self.do_fe_for_both(train)
        test = self.do_fe_for_both(test)
        return train, test

    @staticmethod
    def do_fe_for_both(df):
        for col in pred_cols.USER_GROUPBY_COLS:
            sum_col_name = "user_sum_" + col
            mean_col_name = "user_mean_" + col
            max_col_name = "user_max_" + col
            df[sum_col_name] = df.groupby("id")[col].transform("sum")
            df[mean_col_name] = df.groupby("id")[col].transform("mean")
            df[max_col_name] = df.groupby("id")[col].transform("max")
        return df


