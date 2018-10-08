import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)


import numpy as np
import pandas as pd
from rossman.common import pocket_timer, pocket_logger, pocket_file_io, path_const

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train = csv_io.get_file(path_const.ORG_TRAIN)
print(train.describe())
timer.time("read csv")
print("-"*40)

train["Date"] = pd.to_datetime(train["Date"])
train["Month"] = train["Date"].dt.month
train["Week"] = train["Date"].dt.week
train["Day"] = train["Date"].dt.day

print(train.head())
print(train.info())
print(train.describe())

csv_io.output_csv(train, path_const.TRAIN1)
timer.time("output csv")
