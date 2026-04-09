from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_NAME = "flowvol_120_v1"
FACTOR_DESCRIPTION = (
    "120s signed flow-momentum (ret1s*buy-imbalance) normalized by 120s return volatility"
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

    vol_safe = volume.replace(0, np.nan)
    buy_imbalance = (2.0 * taker_buy_base - volume) / vol_safe

    signed_flow_momentum_120 = (ret_1s * buy_imbalance).rolling(
        window=120, min_periods=40
    ).sum()
    ret_vol_120 = ret_1s.rolling(window=120, min_periods=40).std()

    factor = signed_flow_momentum_120 / (ret_vol_120 + 1e-12)
    return factor.astype("float64").rename(FACTOR_NAME)
