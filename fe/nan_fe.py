from rstudio.common import pred_cols


class NanFe:

    def do_fe(self, train, test):
        train = self.do_fe_for_both(train)
        test = self.do_fe_for_both(test)
        return train, test

    @staticmethod
    def do_fe_for_both(df):
        for col in pred_cols.HAS_NAN_COLS:
            new_col_name = "is_nan_" + col
            df[new_col_name] = df[col].isna()
        return df


