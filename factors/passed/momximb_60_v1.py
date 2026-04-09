from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_NAME = "momximb_60_v1"
FACTOR_DESCRIPTION = (
    "60s log-momentum multiplied by 60s taker-buy volume imbalance mean"
)
REQUIRED_COLUMNS = ["close", "volume", "taker_buy_base_asset_volume"]


def compute_factor(df: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    volume = pd.to_numeric(df["volume"], errors="coerce").astype("float64")
    taker_buy_base = pd.to_numeric(
        df["taker_buy_base_asset_volume"], errors="coerce"
    ).astype("float64")

    log_close = np.log(close.replace(0, np.nan))
    ret_1s = log_close.diff()
    mom_60 = ret_1s.rolling(window=60, min_periods=20).sum()

    vol_safe = volume.replace(0, np.nan)
    buy_imbalance = (2.0 * taker_buy_base - volume) / vol_safe
    imbalance_60 = buy_imbalance.rolling(window=60, min_periods=20).mean()

    factor = mom_60 * imbalance_60
    return factor.astype("float64").rename(FACTOR_NAME)
