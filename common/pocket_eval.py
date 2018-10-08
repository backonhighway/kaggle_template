import pandas as pd
import numpy as np
from . import pocket_logger
from sklearn import metrics


class GoldenEval:
    def __init__(self, pred_cols):
        self.pred_cols = pred_cols
        self.logger = pocket_logger.get_my_logger()

    def evaluate(self, model, holdout: pd.DataFrame):
        holdout_x = holdout[self.pred_cols]
        y_pred = model.predict(holdout_x)
        y_true = holdout["totals.transactionRevenue"]
        self.print_score(y_true, y_pred, "default_score= ")
        clipped_pred = np.clip(y_pred, 0, None)
        self.print_score(y_true, clipped_pred, "clipped_score= ")

        holdout["y_pred"] = np.expm1(clipped_pred)  # non-clipped could be better
        holdout["y_true"] = np.expm1(holdout["totals.transactionRevenue"])

        use_col = ["y_pred", "y_true"]
        grouped = holdout.groupby("fullVisitorId")[use_col].sum().reset_index()
        grouped["y_pred"] = np.clip(grouped["y_pred"], 0.0, None)
        y_pred = np.log1p(grouped["y_pred"])
        y_true = np.log1p(grouped["y_true"])
        self.print_score(y_true, y_pred, "summed_score= ")

    def print_score(self, y_true, y_pred, text="score="):
        score = metrics.mean_squared_error(y_pred, y_true) ** 0.5
        print_text = text + str(score)
        print(print_text)
        self.logger.info(print_text)