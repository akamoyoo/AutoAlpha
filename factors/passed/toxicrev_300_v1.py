from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_NAME = "toxicrev_300_v1"
FACTOR_DESCRIPTION = (
    "300s order-flow toxicity (abs quote taker-imbalance mean) times 300s reversal momentum"
)
REQUIRED_COLUMNS = ["close", "quote_asset_volume", "taker_buy_quote_asset_volume"]


def compute_factor(df: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    quote_volume = pd.to_numeric(df["quote_asset_volume"], errors="coerce").astype(
        "float64"
    )
    taker_buy_quote = pd.to_numeric(
        df["taker_buy_quote_asset_volume"], errors="coerce"
    ).astype("float64")

    log_close = np.log(close.replace(0, np.nan))
    ret_1s = log_close.diff()

    quote_vol_safe = quote_volume.replace(0, np.nan)
    quote_buy_imbalance = (2.0 * taker_buy_quote - quote_volume) / quote_vol_safe

    toxicity_300 = quote_buy_imbalance.abs().rolling(window=300, min_periods=100).mean()
    reversal_300 = -ret_1s.rolling(window=300, min_periods=100).sum()

    factor = toxicity_300 * reversal_300
    return factor.astype("float64").rename(FACTOR_NAME)
