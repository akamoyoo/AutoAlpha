from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_NAME = "toxicrev_3600_v1"
FACTOR_DESCRIPTION = (
    "3600s order-flow toxicity (abs quote taker-imbalance mean) times 3600s reversal momentum"
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

    toxicity_3600 = quote_buy_imbalance.abs().rolling(window=3600, min_periods=1200).mean()
    reversal_3600 = -ret_1s.rolling(window=3600, min_periods=1200).sum()

    factor = toxicity_3600 * reversal_3600
    return factor.astype("float64").rename(FACTOR_NAME)
