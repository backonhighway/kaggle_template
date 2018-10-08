PRED_COL1 = [
]

CAT_COLS = [
]

HAS_NAN_COLS = [
]
PRED_NAN_COLS = [ "is_nan_" + col for col in HAS_NAN_COLS]

TRAIN_DROP_COL = [
]
TEST_DROP_COL = TRAIN_DROP_COL.copy()
TEST_DROP_COL.remove("trafficSource.campaignCode")

LAG_COLS = [
]
PRED_LAG_COLS = ["lag_tar_" + col for col in LAG_COLS] + ["lag_hits_" + col for col in LAG_COLS]

USER_GROUPBY_COLS = [
]
PRED_USER_SUM_COLS = ["user_sum_" + col for col in USER_GROUPBY_COLS]
PRED_USER_MEAN_COLS = ["user_mean_" + col for col in USER_GROUPBY_COLS]
PRED_USER_MAX_COLS = ["user_max_" + col for col in USER_GROUPBY_COLS]


TRAIN_ALL_COLS = PRED_COL1 + PRED_NAN_COLS + PRED_LAG_COLS + \
                 PRED_USER_MEAN_COLS + PRED_USER_SUM_COLS + PRED_USER_MAX_COLS


