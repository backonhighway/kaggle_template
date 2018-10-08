from datetime import datetime


class DateTimeFe:

    def do_fe(self, train, test):
        train = self.do_fe_for_both(train)
        test = self.do_fe_for_both(test)
        return train, test

    @staticmethod
    def do_fe_for_both(df):
        df["datetime"] = df["visitStartTime"].apply(datetime.fromtimestamp)
        df["visit_year"] = df["datetime"].dt.year
        df["visit_month"] = df["datetime"].dt.month
        df["visit_week"] = df["datetime"].dt.week
        df["visit_day"] = df["datetime"].dt.day
        df["visit_hour"] = df["datetime"].dt.hour
        df["visit_day_of_week"] = df["datetime"].dt.dayofweek
        return df
