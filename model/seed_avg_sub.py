import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import numpy as np
import pandas as pd
from rstudio.common import pocket_timer, pocket_logger, pocket_file_io, path_const, pred_cols
from rstudio.common import pocket_lgb, holdout_splitter

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()
predict_col = pred_cols.TRAIN_ALL_COLS

train = csv_io.read_file(path_const.TRAIN1)
test = csv_io.read_file(path_const.TEST1)
timer.time("load csv")
# train["fullVisitorId"] = train["fullVisitorId"].astype("float")

train_x, train_y = train[predict_col], train["target"]
test_x = test[predict_col]
timer.time("prepare train in ")

test["y_pred"] = 0
bagging_num = 4
set_timer = pocket_timer.GoldenTimer(logger)
for bagging_index in range(bagging_num):
    random_state = 71 * bagging_index
    lgb = pocket_lgb.GoldenLgb(seed=random_state)
    model = lgb.train_no_holdout(train_x, train_y)
    y_pred = model.predict(test_x)
    test["y_pred"] = test["y_pred"] + y_pred

    if bagging_index == 0:
        lgb.show_feature_importance(model)
    set_timer.time("done one set in")

timer.time("end train in ")

test["y_pred"] = test["y_pred"] / bagging_num
print(test["y_pred"].describe())
sub = pd.DataFrame()
sub["id"] = test["id"]
sub["y_pred"] = np.clip(test["y_pred"], 0.0, None)
csv_io.output_csv(sub, path_const.SUB)
timer.time("done sub in ")
