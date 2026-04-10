#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────

@dataclass
class Thresholds:
    # Basic quality
    min_coverage: float = 0.8
    min_unique_ratio: float = 0.001
    min_std: float = 1e-8

    # Predictive power
    min_abs_spearman: float = 0.005
    min_abs_pearson: float = 0.005
    min_quantile_spread: float = 0.0

    # IC stability
    min_abs_icir: float = 0.05          # |IC mean / IC std|
    min_ic_positive_ratio: float = 0.0  # fraction of rolling windows with IC > 0 (0 = off)

    # Decay: IC should not collapse too fast
    max_ic_decay_ratio: float = 10.0    # IC(lag=1) / IC(lag=N); > threshold means fast decay

    # Distribution quality
    max_abs_skew: float = 10.0          # winsorised factor skewness ceiling
    max_kurtosis: float = 100.0         # excess kurtosis ceiling

    # Turnover / tradability
    min_autocorr_lag1: float = 0.0      # rank autocorrelation at lag-1 (0 = off)
    max_rank_turnover: float = 1.0      # mean abs rank change per bar (1 = off)

    # Layer testing
    min_layer_monotonicity: float = 0.0 # fraction of adjacent-layer pairs where return increases; 0 = off

    # Multicollinearity
    max_abs_corr_with_passed: float = 0.9


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def normalize_key_part(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def data_fingerprint(data_file: Path) -> str:
    stat = data_file.stat()
    raw = f"{data_file.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def factor_source_fingerprint(factor_file: Path) -> str:
    return hashlib.sha256(factor_file.read_bytes()).hexdigest()[:16]


def build_cache_path(cache_dir: Path, data_file: Path, target_col: str, factor_file: Path) -> Path:
    return (
        cache_dir
        / data_fingerprint(data_file)
        / normalize_key_part(target_col)
        / f"{factor_file.stem}__{factor_source_fingerprint(factor_file)}.parquet"
    )


def load_cached_factor(cache_path: Path, expected_index: pd.Index, rebuild_cache: bool) -> pd.Series | None:
    if rebuild_cache or not cache_path.exists():
        return None
    try:
        cached_df = pd.read_parquet(cache_path)
        if cached_df.shape[1] == 0:
            return None
        factor = pd.to_numeric(cached_df.iloc[:, 0], errors="coerce").astype("float64")
        if len(factor) != len(expected_index):
            return None
        factor.index = expected_index
        return factor
    except Exception:
        return None


def save_cached_factor(cache_path: Path, factor: pd.Series) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    factor.rename("factor").to_frame().to_parquet(cache_path)


def load_module_from_file(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import factor file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return None


def winsorise(s: pd.Series, pct: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


# ─────────────────────────────────────────────
# Basic metrics (coverage, std, pearson, spearman, quantile spread)
# ─────────────────────────────────────────────

def calc_basic_metrics(factor: pd.Series, target: pd.Series) -> dict[str, Any]:
    valid = factor.notna()
    coverage = float(valid.mean())

    f = factor[valid]
    t = target[valid]

    std = safe_float(f.std(ddof=1))
    unique_ratio = safe_float(f.nunique(dropna=True) / max(len(f), 1))

    pearson = spearman = None
    if len(f) > 2 and t.notna().sum() > 2:
        pearson = safe_float(f.corr(t, method="pearson"))
        spearman = safe_float(f.corr(t, method="spearman"))

    quantile_spread = None
    try:
        joined = pd.DataFrame({"factor": f, "target": t}).dropna()
        if len(joined) > 100:
            bins = pd.qcut(joined["factor"], q=10, labels=False, duplicates="drop")
            grouped = joined.groupby(bins)["target"].mean()
            if len(grouped) >= 2:
                quantile_spread = safe_float(grouped.iloc[-1] - grouped.iloc[0])
    except Exception:
        pass

    return {
        "coverage": coverage,
        "unique_ratio": unique_ratio,
        "std": std,
        "pearson": pearson,
        "spearman": spearman,
        "quantile_spread": quantile_spread,
        "valid_count": int(valid.sum()),
        "total_count": int(len(factor)),
    }


# ─────────────────────────────────────────────
# IC series: rolling Spearman IC, ICIR
# ─────────────────────────────────────────────

def calc_ic_series(
    factor: pd.Series,
    target: pd.Series,
    window: int = 2400,
    min_obs: int = 600,
) -> dict[str, Any]:
    """
    Rolling Spearman IC between factor and target.

    Returns ic_mean, ic_std, icir, ic_positive_ratio, and a
    condensed ic_series snapshot (every 'sample_step' values)
    for downstream plotting.
    """
    joined = pd.DataFrame({"f": factor, "t": target}).dropna()
    if len(joined) < min_obs:
        return {
            "ic_mean": None, "ic_std": None, "icir": None,
            "ic_positive_ratio": None, "ic_series_sample": [],
        }

    # Vectorised rolling rank-correlation approximation:
    # rank both series inside each window, then pearson on ranks ≈ spearman
    def _rolling_spearman(df: pd.DataFrame, w: int) -> pd.Series:
        mp = min(min_obs, w)
        # rank the full series, then compute rolling pearson on ranks
        rf = df["f"].rank()
        rt = df["t"].rank()
        cov = (rf * rt).rolling(w, min_periods=mp).mean() \
              - rf.rolling(w, min_periods=mp).mean() * rt.rolling(w, min_periods=mp).mean()
        std_f = rf.rolling(w, min_periods=mp).std()
        std_t = rt.rolling(w, min_periods=mp).std()
        return (cov / (std_f * std_t + 1e-12)).rename("ic")

    ic = _rolling_spearman(joined, window).dropna()
    if len(ic) < 2:
        return {
            "ic_mean": None, "ic_std": None, "icir": None,
            "ic_positive_ratio": None, "ic_series_sample": [],
        }

    ic_mean = safe_float(ic.mean())
    ic_std  = safe_float(ic.std())
    icir    = safe_float(ic_mean / ic_std if ic_std and ic_std > 1e-10 else None)
    ic_pos  = safe_float((ic > 0).mean())

    # Downsample for storage (keep at most 200 points)
    step = max(1, len(ic) // 200)
    ic_sample = [safe_float(v) for v in ic.iloc[::step].tolist()]

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "ic_positive_ratio": ic_pos,
        "ic_series_sample": ic_sample,
    }


# ─────────────────────────────────────────────
# IC decay: IC at multiple forward lags
# ─────────────────────────────────────────────

def calc_ic_decay(
    factor: pd.Series,
    target_df: pd.DataFrame,          # original df with close/return columns
    lags: tuple[int, ...] = (1, 5, 10, 30, 60, 120, 300),
    return_col: str = "log_return",
) -> dict[str, Any]:
    """
    Compute IC(factor, forward_return_at_lag_k) for each lag k.
    Requires a return column in target_df so we can build multi-lag targets.

    decay_ratio = IC(lag=1) / IC(lag=max_lag)  — higher = faster decay.
    half_life   = lag at which IC drops to 50% of IC(lag=1).
    """
    if return_col not in target_df.columns:
        return {"ic_by_lag": {}, "decay_ratio": None, "half_life_lag": None}

    ret = pd.to_numeric(target_df[return_col], errors="coerce").astype("float64")
    results: dict[str, float | None] = {}

    for lag in lags:
        fwd_ret = ret.shift(-lag)           # forward return k bars ahead
        joined = pd.DataFrame({"f": factor, "r": fwd_ret}).dropna()
        if len(joined) < 30:
            results[str(lag)] = None
            continue
        ic = safe_float(joined["f"].corr(joined["r"], method="spearman"))
        results[str(lag)] = ic

    # Decay ratio and half-life
    ic_lag1 = results.get("1")
    decay_ratio = None
    half_life_lag = None

    if ic_lag1 is not None and abs(ic_lag1) > 1e-8:
        last_key = str(lags[-1])
        ic_last = results.get(last_key)
        if ic_last is not None:
            decay_ratio = safe_float(abs(ic_lag1) / max(abs(ic_last), 1e-12))

        # Half-life: first lag where |IC| <= 0.5 * |IC(lag=1)|
        threshold = 0.5 * abs(ic_lag1)
        for lag in lags[1:]:
            ic_k = results.get(str(lag))
            if ic_k is not None and abs(ic_k) <= threshold:
                half_life_lag = lag
                break

    return {
        "ic_by_lag": results,
        "decay_ratio": decay_ratio,
        "half_life_lag": half_life_lag,
    }


# ─────────────────────────────────────────────
# Distribution quality: skewness, kurtosis
# ─────────────────────────────────────────────

def calc_distribution(factor: pd.Series) -> dict[str, Any]:
    """
    Skewness and excess kurtosis of the (winsorised) factor.
    Also flags heavy tails via Jarque-Bera p-value.
    """
    f = factor.dropna()
    if len(f) < 10:
        return {"skew": None, "kurtosis_excess": None, "jb_pvalue": None, "winsor_skew": None}

    skew     = safe_float(scipy_stats.skew(f))
    kurt     = safe_float(scipy_stats.kurtosis(f, fisher=True))   # excess kurtosis
    jb_stat, jb_p = scipy_stats.jarque_bera(f)
    jb_pvalue = safe_float(jb_p)

    # Winsorised skewness (robust to outliers)
    fw = winsorise(f)
    winsor_skew = safe_float(scipy_stats.skew(fw))

    return {
        "skew": skew,
        "kurtosis_excess": kurt,
        "jb_pvalue": jb_pvalue,       # low p → non-normal; informational only
        "winsor_skew": winsor_skew,
    }


# ─────────────────────────────────────────────
# Turnover & autocorrelation
# ─────────────────────────────────────────────

def calc_turnover(factor: pd.Series) -> dict[str, Any]:
    """
    Rank autocorrelation at lags 1, 5, 10 and mean absolute rank change.
    High turnover = high transaction costs.
    """
    ranked = factor.rank(pct=True)
    ac1  = safe_float(ranked.autocorr(lag=1))
    ac5  = safe_float(ranked.autocorr(lag=5))
    ac10 = safe_float(ranked.autocorr(lag=10))
    rank_turnover = safe_float(ranked.diff().abs().mean())

    return {
        "autocorr_lag1":  ac1,
        "autocorr_lag5":  ac5,
        "autocorr_lag10": ac10,
        "rank_turnover":  rank_turnover,
    }


# ─────────────────────────────────────────────
# Layer (quantile) testing
# ─────────────────────────────────────────────

def calc_layer_test(
    factor: pd.Series,
    target: pd.Series,
    n_layers: int = 10,
) -> dict[str, Any]:
    """
    Split factor into n_layers quantile buckets.
    Compute mean forward return per layer.

    Returns:
      layer_returns        : list of mean return per layer (low → high factor)
      monotonicity_score   : fraction of adjacent pairs where return[i+1] > return[i]
                             (1.0 = perfectly monotone, 0.5 = random, 0.0 = inverted)
      top_minus_bottom     : return spread between top and bottom layer
      long_short_sharpe    : annualised Sharpe of a daily long-top / short-bottom strategy
                             (assumes bars are 1-second; annualisation = sqrt(86400*252))
    """
    joined = pd.DataFrame({"f": factor, "t": target}).dropna()
    if len(joined) < n_layers * 20:
        return {
            "layer_returns": [],
            "monotonicity_score": None,
            "top_minus_bottom": None,
            "long_short_sharpe": None,
        }

    try:
        bins = pd.qcut(joined["f"], q=n_layers, labels=False, duplicates="drop")
    except Exception:
        return {
            "layer_returns": [],
            "monotonicity_score": None,
            "top_minus_bottom": None,
            "long_short_sharpe": None,
        }

    layer_means = joined.groupby(bins)["t"].mean()
    if len(layer_means) < 2:
        return {
            "layer_returns": [],
            "monotonicity_score": None,
            "top_minus_bottom": None,
            "long_short_sharpe": None,
        }

    layer_ret_list = [safe_float(v) for v in layer_means.tolist()]

    # Monotonicity: fraction of adjacent pairs increasing
    pairs = list(zip(layer_ret_list[:-1], layer_ret_list[1:]))
    mono = safe_float(sum(b > a for a, b in pairs if a is not None and b is not None) / len(pairs))

    top_minus_bottom = safe_float(
        layer_means.iloc[-1] - layer_means.iloc[0]
        if len(layer_means) >= 2 else None
    )

    # Long-short time series Sharpe
    # Assign +1 to top decile, -1 to bottom decile, 0 elsewhere
    ls_sharpe = None
    try:
        top_label    = bins.max()
        bottom_label = bins.min()
        ls_ret = joined["t"].where(bins == top_label, 0) \
               - joined["t"].where(bins == bottom_label, 0)
        ls_ret = ls_ret[bins.isin([top_label, bottom_label])]
        if ls_ret.std() > 1e-10:
            ann_factor = np.sqrt(86400 * 252)   # 1-second bars
            ls_sharpe = safe_float(ls_ret.mean() / ls_ret.std() * ann_factor)
    except Exception:
        pass

    return {
        "layer_returns": layer_ret_list,
        "monotonicity_score": mono,
        "top_minus_bottom": top_minus_bottom,
        "long_short_sharpe": ls_sharpe,
    }


# ─────────────────────────────────────────────
# Regime stability: IC split by volatility regime
# ─────────────────────────────────────────────

def calc_regime_stability(
    factor: pd.Series,
    target: pd.Series,
    target_df: pd.DataFrame,
    vol_col: str | None = None,
    return_col: str = "log_return",
    n_regimes: int = 3,
) -> dict[str, Any]:
    """
    Split the sample into n_regimes volatility regimes (low/mid/high).
    Compute IC per regime to check if the factor works across market conditions.

    regime_ic_min  : worst IC across regimes
    regime_ic_std  : cross-regime IC standard deviation (low = stable)
    """
    # Build a volatility proxy
    ret_col = vol_col if (vol_col and vol_col in target_df.columns) else return_col
    if ret_col not in target_df.columns:
        return {"regime_ic": {}, "regime_ic_min": None, "regime_ic_std": None}

    ret = pd.to_numeric(target_df[ret_col], errors="coerce")
    rolling_vol = ret.rolling(300, min_periods=60).std()

    joined = pd.DataFrame({"f": factor, "t": target, "vol": rolling_vol}).dropna()
    if len(joined) < n_regimes * 50:
        return {"regime_ic": {}, "regime_ic_min": None, "regime_ic_std": None}

    try:
        regime_labels = pd.qcut(joined["vol"], q=n_regimes, labels=False, duplicates="drop")
    except Exception:
        return {"regime_ic": {}, "regime_ic_min": None, "regime_ic_std": None}

    regime_ic: dict[str, float | None] = {}
    for r in sorted(regime_labels.unique()):
        mask = regime_labels == r
        sub = joined[mask]
        if len(sub) < 10:
            regime_ic[str(r)] = None
            continue
        ic = safe_float(sub["f"].corr(sub["t"], method="spearman"))
        regime_ic[str(r)] = ic

    valid_ics = [v for v in regime_ic.values() if v is not None]
    regime_ic_min = safe_float(min(valid_ics, key=abs)) if valid_ics else None
    regime_ic_std = safe_float(np.std(valid_ics)) if len(valid_ics) >= 2 else None

    return {
        "regime_ic": regime_ic,
        "regime_ic_min": regime_ic_min,
        "regime_ic_std": regime_ic_std,
    }


# ─────────────────────────────────────────────
# Threshold evaluation
# ─────────────────────────────────────────────

def evaluate_thresholds(metrics: dict[str, Any], th: Thresholds) -> list[str]:
    reasons: list[str] = []

    def _check(condition: bool, msg: str):
        if condition:
            reasons.append(msg)

    # Basic
    _check(
        metrics["coverage"] < th.min_coverage,
        f"coverage too low: {metrics['coverage']:.4f} < {th.min_coverage:.4f}",
    )
    _check(
        metrics["unique_ratio"] is None or metrics["unique_ratio"] < th.min_unique_ratio,
        f"unique_ratio too low: {metrics['unique_ratio']} < {th.min_unique_ratio:.6f}",
    )
    _check(
        metrics["std"] is None or metrics["std"] < th.min_std,
        f"std too low: {metrics['std']} < {th.min_std:.2e}",
    )

    pearson_abs  = abs(metrics["pearson"])  if metrics["pearson"]  is not None else 0.0
    spearman_abs = abs(metrics["spearman"]) if metrics["spearman"] is not None else 0.0
    _check(
        pearson_abs < th.min_abs_pearson and spearman_abs < th.min_abs_spearman,
        f"predictive correlation too weak: |pearson|={pearson_abs:.6f}, |spearman|={spearman_abs:.6f}",
    )

    q_spread = metrics.get("quantile_spread")
    if q_spread is not None:
        _check(
            abs(q_spread) < th.min_quantile_spread,
            f"quantile spread too weak: {abs(q_spread):.6f} < {th.min_quantile_spread:.6f}",
        )

    # IC stability
    icir = metrics.get("icir")
    _check(
        icir is not None and abs(icir) < th.min_abs_icir,
        f"ICIR too weak: {icir:.4f} < {th.min_abs_icir:.4f}" if icir is not None else "",
    )

    ic_pos = metrics.get("ic_positive_ratio")
    if th.min_ic_positive_ratio > 0 and ic_pos is not None:
        _check(
            ic_pos < th.min_ic_positive_ratio,
            f"IC positive ratio too low: {ic_pos:.3f} < {th.min_ic_positive_ratio:.3f}",
        )

    # Decay
    decay_ratio = metrics.get("decay_ratio")
    if decay_ratio is not None:
        _check(
            decay_ratio > th.max_ic_decay_ratio,
            f"IC decays too fast: decay_ratio={decay_ratio:.2f} > {th.max_ic_decay_ratio:.2f}",
        )

    # Distribution
    skew = metrics.get("winsor_skew")
    if skew is not None:
        _check(
            abs(skew) > th.max_abs_skew,
            f"|skew| too high: {abs(skew):.2f} > {th.max_abs_skew:.2f}",
        )

    kurt = metrics.get("kurtosis_excess")
    if kurt is not None:
        _check(
            kurt > th.max_kurtosis,
            f"excess kurtosis too high: {kurt:.2f} > {th.max_kurtosis:.2f}",
        )

    # Turnover
    ac1 = metrics.get("autocorr_lag1")
    if th.min_autocorr_lag1 > 0 and ac1 is not None:
        _check(
            ac1 < th.min_autocorr_lag1,
            f"autocorr_lag1 too low: {ac1:.4f} < {th.min_autocorr_lag1:.4f}",
        )

    turnover = metrics.get("rank_turnover")
    if th.max_rank_turnover < 1.0 and turnover is not None:
        _check(
            turnover > th.max_rank_turnover,
            f"rank_turnover too high: {turnover:.4f} > {th.max_rank_turnover:.4f}",
        )

    # Layer monotonicity
    mono = metrics.get("monotonicity_score")
    if th.min_layer_monotonicity > 0 and mono is not None:
        _check(
            mono < th.min_layer_monotonicity,
            f"layer monotonicity too low: {mono:.3f} < {th.min_layer_monotonicity:.3f}",
        )

    return [r for r in reasons if r]   # drop empty strings


# ─────────────────────────────────────────────
# Multicollinearity check
# ─────────────────────────────────────────────

def compute_abs_corr(a: pd.Series, b: pd.Series) -> float | None:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 3:
        return None
    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1], method="pearson")
    return safe_float(abs(corr) if corr is not None else None)


def evaluate_multicollinearity(
    candidate_factor: pd.Series,
    df: pd.DataFrame,
    data_file: Path,
    target_col: str,
    passed_dir: Path,
    max_abs_corr_with_passed: float,
    cache_dir: Path | None,
    use_cache: bool,
    rebuild_cache: bool,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    details: list[dict[str, Any]] = []
    reasons: list[str] = []
    cache_stats = {"hits": 0, "misses": 0, "writes": 0}

    if not passed_dir.exists():
        return details, reasons, cache_stats

    for py_file in sorted(passed_dir.glob("*.py")):
        try:
            cache_path = None
            used_cache = False
            module = load_module_from_file(py_file)
            if not hasattr(module, "compute_factor"):
                continue

            other_factor: pd.Series | None = None
            if use_cache and cache_dir is not None:
                cache_path = build_cache_path(
                    cache_dir=cache_dir,
                    data_file=data_file,
                    target_col=target_col,
                    factor_file=py_file,
                )
                other_factor = load_cached_factor(
                    cache_path=cache_path,
                    expected_index=df.index,
                    rebuild_cache=rebuild_cache,
                )
                if other_factor is not None:
                    cache_stats["hits"] += 1
                    used_cache = True
                else:
                    cache_stats["misses"] += 1

            if other_factor is None:
                other_factor = module.compute_factor(df)
                if use_cache and cache_path is not None and isinstance(other_factor, pd.Series):
                    save_cached_factor(
                        cache_path,
                        pd.to_numeric(other_factor, errors="coerce").astype("float64"),
                    )
                    cache_stats["writes"] += 1

            if not isinstance(other_factor, pd.Series):
                continue

            other_factor = pd.to_numeric(other_factor, errors="coerce").astype("float64")
            corr_abs = compute_abs_corr(candidate_factor, other_factor)
            factor_name = getattr(module, "FACTOR_NAME", py_file.stem)

            details.append({
                "other_factor_file": py_file.name,
                "other_factor_name": factor_name,
                "abs_corr": corr_abs,
                "cache_hit": used_cache,
            })

            if corr_abs is not None and corr_abs > max_abs_corr_with_passed:
                reasons.append(
                    f"too collinear with passed factor {py_file.name}: "
                    f"|corr|={corr_abs:.6f} > {max_abs_corr_with_passed:.6f}"
                )
        except Exception as e:
            details.append({"other_factor_file": py_file.name, "error": str(e)})

    return details, reasons, cache_stats


# ─────────────────────────────────────────────
# Target column resolution
# ─────────────────────────────────────────────

def resolve_target_col(df: pd.DataFrame, target_col: str) -> tuple[str, str | None]:
    if target_col in df.columns:
        return target_col, None
    fallbacks = ["log_return", "logret_1h", "target", "y"]
    fallback = next((c for c in fallbacks if c in df.columns), None)
    if fallback is None:
        raise KeyError(
            f"target column '{target_col}' not found. available: {list(df.columns)}"
        )
    return fallback, f"requested '{target_col}' not found; fallback to '{fallback}'"


def move_factor_file(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = dest_dir / f"{src.stem}_{ts}{src.suffix}"
    shutil.move(str(src), str(dest))
    return dest


# ─────────────────────────────────────────────
# Main evaluation entry point
# ─────────────────────────────────────────────

def evaluate_single_factor(
    factor_file: Path,
    data_file: Path,
    target_col: str,
    th: Thresholds,
    inbox_dir: Path,
    passed_dir: Path,
    rejected_dir: Path,
    reports_dir: Path,
    cache_dir: Path | None,
    use_cache: bool,
    rebuild_cache: bool,
    ic_window: int = 2400,
    ic_decay_lags: tuple[int, ...] = (1, 5, 10, 30, 60, 120, 300),
    n_layers: int = 10,
    df: pd.DataFrame | None = None,   # pre-loaded df (batch optimisation)
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report: dict[str, Any] = {
        "timestamp": timestamp,
        "factor_file": str(factor_file),
        "status": "rejected",
        "reason": [],
    }

    try:
        if not factor_file.exists():
            raise FileNotFoundError(f"factor file not found: {factor_file}")

        if df is None:
            df = pd.read_parquet(data_file)

        target_col, fallback_note = resolve_target_col(df, target_col)
        if fallback_note:
            report["target_col_fallback"] = fallback_note

        module = load_module_from_file(factor_file)
        if not hasattr(module, "compute_factor"):
            raise AttributeError("factor file must define compute_factor(df)")

        required_cols = getattr(module, "REQUIRED_COLUMNS", [])
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"missing required columns: {missing_cols}")

        factor_raw = module.compute_factor(df)
        if not isinstance(factor_raw, pd.Series):
            raise TypeError("compute_factor(df) must return pandas.Series")

        factor = pd.to_numeric(factor_raw, errors="coerce").astype("float64")
        if len(factor) != len(df):
            raise ValueError(
                f"factor length mismatch: factor={len(factor)}, data={len(df)}"
            )

        factor.index = df.index
        target = pd.to_numeric(df[target_col], errors="coerce").astype("float64")

        # ── Run all metric modules ────────────────────────────────────────
        basic       = calc_basic_metrics(factor, target)
        ic_stats    = calc_ic_series(factor, target, window=ic_window)
        ic_decay    = calc_ic_decay(factor, df, lags=ic_decay_lags, return_col=target_col)
        dist_stats  = calc_distribution(factor)
        turnover    = calc_turnover(factor)
        layer_stats = calc_layer_test(factor, target, n_layers=n_layers)
        regime      = calc_regime_stability(factor, target, df, return_col=target_col)

        # Flatten all metrics into one dict for threshold evaluation
        all_metrics: dict[str, Any] = {
            **basic,
            **ic_stats,
            **ic_decay,
            **dist_stats,
            **turnover,
            **layer_stats,
        }

        threshold_reasons = evaluate_thresholds(all_metrics, th)

        # ── Cache candidate factor ────────────────────────────────────────
        if use_cache and cache_dir is not None:
            candidate_cache_path = build_cache_path(
                cache_dir=cache_dir,
                data_file=data_file,
                target_col=target_col,
                factor_file=factor_file,
            )
            if rebuild_cache or not candidate_cache_path.exists():
                save_cached_factor(candidate_cache_path, factor)

        # ── Multicollinearity ─────────────────────────────────────────────
        corr_details, corr_reasons, cache_stats = evaluate_multicollinearity(
            candidate_factor=factor,
            df=df,
            data_file=data_file,
            target_col=target_col,
            passed_dir=passed_dir,
            max_abs_corr_with_passed=th.max_abs_corr_with_passed,
            cache_dir=cache_dir,
            use_cache=use_cache,
            rebuild_cache=rebuild_cache,
        )

        all_reasons = threshold_reasons + corr_reasons
        factor_name = getattr(module, "FACTOR_NAME", factor_file.stem)
        factor_desc = getattr(module, "FACTOR_DESCRIPTION", "")

        report.update({
            "factor_name":      factor_name,
            "factor_description": factor_desc,
            "target_col":       target_col,
            # Structured metric sections
            "metrics": {
                "basic":    basic,
                "ic":       ic_stats,
                "decay":    ic_decay,
                "distribution": dist_stats,
                "turnover": turnover,
                "layers":   layer_stats,
                "regime":   regime,
            },
            "thresholds":        th.__dict__,
            "collinearity_check": corr_details,
            "cache_stats":       cache_stats,
            "reason":            all_reasons,
        })

        if all_reasons:
            dest = move_factor_file(factor_file, rejected_dir)
            report["status"] = "rejected"
            report["moved_to"] = str(dest)
        else:
            dest = move_factor_file(factor_file, passed_dir)
            report["status"] = "passed"
            report["moved_to"] = str(dest)

    except Exception as e:
        report["status"] = "rejected"
        report["reason"] = [f"runtime error: {e}"]
        report["traceback"] = traceback.format_exc(limit=8)
        try:
            if factor_file.exists() and factor_file.parent.resolve() == inbox_dir.resolve():
                dest = move_factor_file(factor_file, rejected_dir)
                report["moved_to"] = str(dest)
        except Exception as move_error:
            report["move_error"] = str(move_error)

    finally:
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_name = f"{factor_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = reports_dir / report_name
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
        report["report_file"] = str(report_path)

    return report


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate factor files and auto-route to passed/rejected."
    )
    parser.add_argument(
        "--data-file",
        default="Data/BTCUSDT/klines_1s_monthly_filled_all_with_logret_1h.parquet",
    )
    parser.add_argument("--target-col", default="log_return")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--factor", help="Single factor .py file path")
    group.add_argument("--batch-inbox", action="store_true",
                       help="Evaluate all .py files under factors/inbox")
    group.add_argument("--build-cache-from-passed", action="store_true",
                       help="Precompute cache for all factors/passed/*.py")

    parser.add_argument("--inbox-dir",    default="factors/inbox")
    parser.add_argument("--passed-dir",   default="factors/passed")
    parser.add_argument("--rejected-dir", default="factors/rejected")
    parser.add_argument("--reports-dir",  default="factors/reports")
    parser.add_argument("--cache-dir",    default="factors/cache")
    parser.add_argument("--no-cache",     action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")

    # IC params
    parser.add_argument("--ic-window",  type=int, default=2400,
                        help="Rolling window size for IC calculation (bars)")
    parser.add_argument("--n-layers",   type=int, default=10,
                        help="Number of quantile layers for layer testing")

    # Thresholds
    parser.add_argument("--min-coverage",              type=float, default=0.8)
    parser.add_argument("--min-unique-ratio",          type=float, default=0.001)
    parser.add_argument("--min-std",                   type=float, default=1e-8)
    parser.add_argument("--min-abs-spearman",          type=float, default=0.005)
    parser.add_argument("--min-abs-pearson",           type=float, default=0.005)
    parser.add_argument("--min-quantile-spread",       type=float, default=0.0)
    parser.add_argument("--min-abs-icir",              type=float, default=0.05)
    parser.add_argument("--min-ic-positive-ratio",     type=float, default=0.0)
    parser.add_argument("--max-ic-decay-ratio",        type=float, default=10.0)
    parser.add_argument("--max-abs-skew",              type=float, default=10.0)
    parser.add_argument("--max-kurtosis",              type=float, default=100.0)
    parser.add_argument("--min-autocorr-lag1",         type=float, default=0.0)
    parser.add_argument("--max-rank-turnover",         type=float, default=1.0)
    parser.add_argument("--min-layer-monotonicity",    type=float, default=0.0)
    parser.add_argument("--max-abs-corr-with-passed",  type=float, default=0.9)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    th = Thresholds(
        min_coverage=args.min_coverage,
        min_unique_ratio=args.min_unique_ratio,
        min_std=args.min_std,
        min_abs_spearman=args.min_abs_spearman,
        min_abs_pearson=args.min_abs_pearson,
        min_quantile_spread=args.min_quantile_spread,
        min_abs_icir=args.min_abs_icir,
        min_ic_positive_ratio=args.min_ic_positive_ratio,
        max_ic_decay_ratio=args.max_ic_decay_ratio,
        max_abs_skew=args.max_abs_skew,
        max_kurtosis=args.max_kurtosis,
        min_autocorr_lag1=args.min_autocorr_lag1,
        max_rank_turnover=args.max_rank_turnover,
        min_layer_monotonicity=args.min_layer_monotonicity,
        max_abs_corr_with_passed=args.max_abs_corr_with_passed,
    )

    inbox_dir    = Path(args.inbox_dir)
    passed_dir   = Path(args.passed_dir)
    rejected_dir = Path(args.rejected_dir)
    reports_dir  = Path(args.reports_dir)
    data_file    = Path(args.data_file)
    cache_dir    = Path(args.cache_dir)
    use_cache    = not args.no_cache

    # ── Build cache from passed ───────────────────────────────────────────
    if args.build_cache_from_passed:
        if not data_file.exists():
            raise FileNotFoundError(f"data file not found: {data_file}")
        if not passed_dir.exists():
            print(f"passed dir not found: {passed_dir}")
            return 0

        df = pd.read_parquet(data_file)
        resolved_target_col, fallback_note = resolve_target_col(df, args.target_col)
        if fallback_note:
            print(f"Target fallback: {fallback_note}")

        passed_files = sorted(passed_dir.glob("*.py"))
        if not passed_files:
            print(f"No factor files found in {passed_dir}")
            return 0

        built = skipped = failed = 0
        for py_file in passed_files:
            try:
                cache_path = build_cache_path(
                    cache_dir=cache_dir,
                    data_file=data_file,
                    target_col=resolved_target_col,
                    factor_file=py_file,
                )
                if cache_path.exists() and not args.rebuild_cache:
                    skipped += 1
                    continue

                module = load_module_from_file(py_file)
                if not hasattr(module, "compute_factor"):
                    failed += 1
                    print(f"[FAILED] {py_file.name}: missing compute_factor(df)")
                    continue

                factor = module.compute_factor(df)
                if not isinstance(factor, pd.Series):
                    failed += 1
                    print(f"[FAILED] {py_file.name}: did not return Series")
                    continue

                factor = pd.to_numeric(factor, errors="coerce").astype("float64")
                if len(factor) != len(df):
                    failed += 1
                    print(f"[FAILED] {py_file.name}: length mismatch")
                    continue

                factor.index = df.index
                save_cached_factor(cache_path, factor)
                built += 1
                print(f"[CACHED] {py_file.name} -> {cache_path}")
            except Exception as e:
                failed += 1
                print(f"[FAILED] {py_file.name}: {e}")

        print(f"Cache build: built={built}, skipped={skipped}, failed={failed}")
        return 0

    # ── Batch or single evaluation ────────────────────────────────────────
    if args.batch_inbox:
        factor_files = sorted(inbox_dir.glob("*.py"))
        if not factor_files:
            print(f"No factor files found in {inbox_dir}")
            return 0
        # Load df once for the entire batch (major speedup)
        shared_df: pd.DataFrame | None = pd.read_parquet(data_file)
    else:
        factor_files = [Path(args.factor)]
        shared_df = None

    summary: dict[str, Any] = {"passed": 0, "rejected": 0, "reports": []}

    for factor_file in factor_files:
        result = evaluate_single_factor(
            factor_file=factor_file,
            data_file=data_file,
            target_col=args.target_col,
            th=th,
            inbox_dir=inbox_dir,
            passed_dir=passed_dir,
            rejected_dir=rejected_dir,
            reports_dir=reports_dir,
            cache_dir=cache_dir,
            use_cache=use_cache,
            rebuild_cache=args.rebuild_cache,
            ic_window=args.ic_window,
            n_layers=args.n_layers,
            df=shared_df,
        )

        summary[result["status"]] += 1
        summary["reports"].append(result.get("report_file"))

        reason_preview = "; ".join(result.get("reason", [])[:2])
        print(f"[{result['status'].upper()}] {factor_file.name} -> {result.get('moved_to', 'N/A')}")
        if reason_preview:
            print(f"  reason: {reason_preview}")

    # Write aggregate summary
    summary_path = reports_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(
        f"Summary: passed={summary['passed']}, rejected={summary['rejected']}, "
        f"reports={len(summary['reports'])} -> {summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())