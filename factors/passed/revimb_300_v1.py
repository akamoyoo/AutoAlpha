from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_NAME = "revimb_300_v1"
FACTOR_DESCRIPTION = (
    "300s reversal pressure: negative 300s log-momentum times 300s taker-buy imbalance mean"
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

    momentum_300 = ret_1s.rolling(window=300, min_periods=100).sum()

    vol_safe = volume.replace(0, np.nan)
    buy_imbalance = (2.0 * taker_buy_base - volume) / vol_safe
    imbalance_300 = buy_imbalance.rolling(window=300, min_periods=100).mean()

    factor = -momentum_300 * imbalance_300
    return factor.astype("float64").rename(FACTOR_NAME)
