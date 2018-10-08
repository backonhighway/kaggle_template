import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import numpy as np
import pandas as pd
from rstudio.common import pocket_timer, pocket_logger, pocket_file_io, path_const, pred_cols
from rstudio.common import pocket_lgb, holdout_splitter, pocket_eval

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()
predict_col = pred_cols.TRAIN_ALL_COLS

train = csv_io.read_file(path_const.TRAIN1)
timer.time("load csv")
# train["fullVisitorId"] = train["fullVisitorId"].astype("float")

splitter = holdout_splitter.GoldenSplitter()
train, holdout = splitter.get_holdout_split(train, 20170701)
train_y, holdout_y = train["target"], holdout["target"]
train_x, holdout_x = train[predict_col], holdout[predict_col]
timer.time("prepare train in ")

lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_direct(train_x, holdout_x, train_y, holdout_y)
lgb.show_feature_importance(model)

evaluator = pocket_eval.GoldenEval(predict_col)
evaluator.evaluate(model, holdout)

timer.time("end train in ")

