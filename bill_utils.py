import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = "next_month_bill"

# Columns the model receives (raw + history columns). Everything else is dropped.
CAT_ONEHOT = ["area", "connection_type", "has_solar"]
CAT_ORDINAL = ["house_type"]
HOUSE_ORDER = [["1BHK", "2BHK", "3BHK", "Independent"]]
NUM_COLS = ["family_size", "num_acs", "avg_temp_c", "humidity_pct", "units_consumed",
            "bill_amount", "prev_bill", "rolling_3m_bill", "bill_change",
            # created by FeatureEngineer:
            "next_month_sin", "next_month_cos", "next_is_summer",
            "units_per_person", "ac_x_temp"]
INPUT_COLS = ["area", "connection_type", "house_type", "has_solar", "family_size", "num_acs",
              "month", "avg_temp_c", "humidity_pct", "units_consumed", "bill_amount",
              "prev_bill", "rolling_3m_bill", "bill_change"]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """History features per customer. They only look BACKWARD in time, so no leakage."""
    df = df.sort_values(["customer_id", "month"]).copy()
    g = df.groupby("customer_id")["bill_amount"]
    df["prev_bill"] = g.shift(1)                                     # last month's bill
    df["rolling_3m_bill"] = g.transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["bill_change"] = df["bill_amount"] - df["prev_bill"]          # trend up / down
    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Row-wise features. Stateless, so it is safe inside the sklearn Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        month_num = pd.to_datetime(X["month"]).dt.month
        next_month = month_num % 12 + 1                     # month we are predicting
        X["next_month_sin"] = np.sin(2 * np.pi * next_month / 12)   # cyclic encoding:
        X["next_month_cos"] = np.cos(2 * np.pi * next_month / 12)   # Dec is close to Jan
        X["next_is_summer"] = next_month.isin([3, 4, 5]).astype(int)
        X["units_per_person"] = X["units_consumed"] / X["family_size"]
        X["ac_x_temp"] = X["num_acs"] * X["avg_temp_c"]     # ACs run harder when hot
        return X
