from sklearn import preprocessing


class CatFe:

    @staticmethod
    def do_fe(train, test):
        cat_cols = [
            "foobar"
        ]
        num_cols = ["hoge"]
        for col in cat_cols:
            print(col)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
            train[col] = lbl.transform(list(train[col].values.astype('str')))
            test[col] = lbl.transform(list(test[col].values.astype('str')))
        # for col in num_cols:
        #     train[col] = train[col].astype(float)
        #     test[col] = test[col].astype(float)

        return train, test

