import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
import numpy as np
import pandas as pd
from rstudio.common import pocket_timer, pocket_logger, pocket_file_io, path_const, pred_cols
from rstudio.fe import cat_fe, datetime_fe, nan_fe, lag_fe, user_fe, domain_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

# train = csv_io.read_file(path_const.PROPER_TRAIN, nrows=1000)
# test = csv_io.read_file(path_const.PROPER_TEST, nrows=1000)
train = csv_io.read_file(path_const.PROPER_TRAIN)
test = csv_io.read_file(path_const.PROPER_TEST)
timer.time("read csv")
print("-"*40)

train["totals.transactionRevenue"].fillna(0, inplace=True)
train["totals.transactionRevenue"] = np.log1p(train["totals.transactionRevenue"])
cat_fer = cat_fe.CatFe()
date_fer = datetime_fe.DateTimeFe()
nan_fer = nan_fe.NanFe()
lag_fer = lag_fe.LagFe()
user_fer = user_fe.UserFe()
domain_fer = domain_fe.DomainFe()
train, test = domain_fer.do_fe(train, test)
train, test = cat_fer.do_fe(train, test)
train, test = date_fer.do_fe(train, test)
train, test = nan_fer.do_fe(train, test)
train, test = lag_fer.do_fe(train, test)
train, test = user_fer.do_fe(train, test)

train = train.drop(pred_cols.TRAIN_DROP_COL, axis=1)
test = test.drop(pred_cols.TEST_DROP_COL, axis=1)
print(train.info())
print(test.info())
csv_io.output_csv(train, path_const.TRAIN1)
csv_io.output_csv(test, path_const.TEST1)


