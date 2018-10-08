import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import numpy as np
import pandas as pd
from rossman.common import pocket_timer, pocket_logger, pocket_file_io, path_const, pred_cols
from rossman.common import pocket_lgb, holdout_splitter

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()
predict_col = pred_cols.PRED_COL1

train = csv_io.get_file(path_const.TRAIN1)
timer.time("load csv")

splitter = holdout_splitter.GoldenSplitter()
train, holdout = splitter.get_holdout_split(train, "2015-06-20")
train_y, holdout_y = train["Sales"], holdout["Sales"]
train_x, holdout_x = train[predict_col], holdout[predict_col]

timer.time("prepare train in ")
lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_direct(train_x, holdout_x, train_y, holdout_y)
lgb.show_feature_importance(model)

timer.time("end train in ")

