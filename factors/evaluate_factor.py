#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Thresholds:
    min_coverage: float = 0.8
    min_unique_ratio: float = 0.001
    min_std: float = 1e-8
    min_abs_spearman: float = 0.005
    min_abs_pearson: float = 0.005
    min_quantile_spread: float = 0.0
    max_abs_corr_with_passed: float = 0.9


def normalize_key_part(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def data_fingerprint(data_file: Path) -> str:
    stat = data_file.stat()
    raw = f"{data_file.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def factor_source_fingerprint(factor_file: Path) -> str:
    payload = factor_file.read_bytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def build_cache_path(
    cache_dir: Path,
    data_file: Path,
    target_col: str,
    factor_file: Path,
) -> Path:
    return (
        cache_dir
        / data_fingerprint(data_file)
        / normalize_key_part(target_col)
        / f"{factor_file.stem}__{factor_source_fingerprint(factor_file)}.parquet"
    )


def load_cached_factor(
    cache_path: Path,
    expected_index: pd.Index,
    rebuild_cache: bool,
) -> pd.Series | None:
    if rebuild_cache or not cache_path.exists():
        return None

    try:
        cached_df = pd.read_parquet(cache_path)
        if cached_df.shape[1] == 0:
            return None
        factor = pd.to_numeric(cached_df.iloc[:, 0], errors="coerce").astype("float64")
        if len(factor) != len(expected_index):
            return None
        # Cache key already binds to data fingerprint; enforce current index for safety.
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
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except Exception:
        return None


def calc_basic_metrics(factor: pd.Series, target: pd.Series) -> dict[str, float | None]:
    valid = factor.notna()
    coverage = float(valid.mean())

    f = factor[valid]
    t = target[valid]

    std = safe_float(f.std(ddof=1))
    unique_ratio = safe_float(f.nunique(dropna=True) / max(len(f), 1))

    if len(f) > 2 and t.notna().sum() > 2:
        pearson = safe_float(f.corr(t, method="pearson"))
        spearman = safe_float(f.corr(t, method="spearman"))
    else:
        pearson = None
        spearman = None

    # Simple monotonic usefulness check via quantile spread
    quantile_spread = None
    try:
        joined = pd.DataFrame({"factor": f, "target": t}).dropna()
        if len(joined) > 100:
            bins = pd.qcut(joined["factor"], q=10, labels=False, duplicates="drop")
            grouped = joined.groupby(bins)["target"].mean()
            if len(grouped) >= 2:
                quantile_spread = safe_float(grouped.iloc[-1] - grouped.iloc[0])
    except Exception:
        quantile_spread = None

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


def evaluate_thresholds(metrics: dict[str, Any], th: Thresholds) -> list[str]:
    reasons: list[str] = []

    if metrics["coverage"] < th.min_coverage:
        reasons.append(
            f"coverage too low: {metrics['coverage']:.4f} < {th.min_coverage:.4f}"
        )

    if metrics["unique_ratio"] is None or metrics["unique_ratio"] < th.min_unique_ratio:
        reasons.append(
            f"unique_ratio too low: {metrics['unique_ratio']} < {th.min_unique_ratio:.6f}"
        )

    if metrics["std"] is None or metrics["std"] < th.min_std:
        reasons.append(f"std too low: {metrics['std']} < {th.min_std:.2e}")

    pearson_abs = abs(metrics["pearson"]) if metrics["pearson"] is not None else 0.0
    spearman_abs = abs(metrics["spearman"]) if metrics["spearman"] is not None else 0.0

    if pearson_abs < th.min_abs_pearson and spearman_abs < th.min_abs_spearman:
        reasons.append(
            "predictive correlation too weak: "
            f"|pearson|={pearson_abs:.6f}, |spearman|={spearman_abs:.6f}"
        )

    q_spread_abs = (
        abs(metrics["quantile_spread"])
        if metrics["quantile_spread"] is not None
        else None
    )
    if (
        q_spread_abs is not None
        and q_spread_abs < th.min_quantile_spread
    ):
        reasons.append(
            f"quantile spread too weak: {q_spread_abs:.6f} < {th.min_quantile_spread:.6f}"
        )

    return reasons


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
            details.append(
                {
                    "other_factor_file": py_file.name,
                    "other_factor_name": factor_name,
                    "abs_corr": corr_abs,
                    "cache_hit": used_cache,
                }
            )

            if corr_abs is not None and corr_abs > max_abs_corr_with_passed:
                reasons.append(
                    "too collinear with passed factor "
                    f"{py_file.name}: |corr|={corr_abs:.6f} > {max_abs_corr_with_passed:.6f}"
                )
        except Exception as e:
            details.append(
                {
                    "other_factor_file": py_file.name,
                    "error": str(e),
                }
            )

    return details, reasons, cache_stats


def resolve_target_col(df: pd.DataFrame, target_col: str) -> tuple[str, str | None]:
    if target_col in df.columns:
        return target_col, None

    fallback_candidates = ["log_return", "logret_1h", "target", "y"]
    fallback = next((c for c in fallback_candidates if c in df.columns), None)
    if fallback is None:
        raise KeyError(
            f"target column '{target_col}' not found in data columns: {list(df.columns)}"
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

        if factor_file.parent.resolve() != inbox_dir.resolve() and factor_file.parent.resolve() != passed_dir.resolve():
            # Allow direct path, but warn in report
            report["path_warning"] = "factor file not in inbox/passed; proceeding anyway"

        df = pd.read_parquet(data_file)

        target_col, fallback_note = resolve_target_col(df, target_col)
        if fallback_note is not None:
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

        metrics = calc_basic_metrics(factor, target)
        threshold_reasons = evaluate_thresholds(metrics, th)

        if use_cache and cache_dir is not None:
            candidate_cache_path = build_cache_path(
                cache_dir=cache_dir,
                data_file=data_file,
                target_col=target_col,
                factor_file=factor_file,
            )
            if rebuild_cache or not candidate_cache_path.exists():
                save_cached_factor(candidate_cache_path, factor)

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

        report.update(
            {
                "factor_name": factor_name,
                "factor_description": factor_desc,
                "target_col": target_col,
                "metrics": metrics,
                "thresholds": th.__dict__,
                "collinearity_check": corr_details,
                "cache_stats": cache_stats,
                "reason": all_reasons,
            }
        )

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
        # If file still exists and is in inbox, move to rejected for triage consistency
        try:
            if factor_file.exists() and factor_file.parent.resolve() == inbox_dir.resolve():
                dest = move_factor_file(factor_file, rejected_dir)
                report["moved_to"] = str(dest)
        except Exception as move_error:
            report["move_error"] = str(move_error)

    finally:
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_name = (
            f"{factor_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        report_path = reports_dir / report_name
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        report["report_file"] = str(report_path)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one factor file and auto-route to passed/rejected."
    )
    parser.add_argument(
        "--data-file",
        default="Data/BTCUSDT/klines_1s_monthly_filled_all_with_logret_1h.parquet",
        help="Path to parquet data file",
    )
    parser.add_argument(
        "--target-col",
        default="log_return",
        help="Target column name in parquet",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--factor", help="Single factor .py file path to evaluate")
    group.add_argument(
        "--batch-inbox",
        action="store_true",
        help="Evaluate all .py files under factors/inbox",
    )
    group.add_argument(
        "--build-cache-from-passed",
        action="store_true",
        help="Compute and persist cache files for all .py under factors/passed",
    )

    parser.add_argument("--inbox-dir", default="factors/inbox")
    parser.add_argument("--passed-dir", default="factors/passed")
    parser.add_argument("--rejected-dir", default="factors/rejected")
    parser.add_argument("--reports-dir", default="factors/reports")
    parser.add_argument("--cache-dir", default="factors/cache")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable factor cache read/write",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recompute cache even if cache files already exist",
    )

    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--min-unique-ratio", type=float, default=0.001)
    parser.add_argument("--min-std", type=float, default=1e-8)
    parser.add_argument("--min-abs-spearman", type=float, default=0.005)
    parser.add_argument("--min-abs-pearson", type=float, default=0.005)
    parser.add_argument("--min-quantile-spread", type=float, default=0.0)
    parser.add_argument("--max-abs-corr-with-passed", type=float, default=0.9)

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
        max_abs_corr_with_passed=args.max_abs_corr_with_passed,
    )

    inbox_dir = Path(args.inbox_dir)
    passed_dir = Path(args.passed_dir)
    rejected_dir = Path(args.rejected_dir)
    reports_dir = Path(args.reports_dir)
    data_file = Path(args.data_file)
    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache

    factor_files: list[Path]
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

        built = 0
        skipped = 0
        failed = 0
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
                    print(f"[FAILED] {py_file.name}: compute_factor did not return Series")
                    continue

                factor = pd.to_numeric(factor, errors="coerce").astype("float64")
                if len(factor) != len(df):
                    failed += 1
                    print(
                        f"[FAILED] {py_file.name}: length mismatch factor={len(factor)} data={len(df)}"
                    )
                    continue

                factor.index = df.index
                save_cached_factor(cache_path, factor)
                built += 1
                print(f"[CACHED] {py_file.name} -> {cache_path}")
            except Exception as e:
                failed += 1
                print(f"[FAILED] {py_file.name}: {e}")

        print(
            f"Cache build summary: built={built}, skipped={skipped}, failed={failed}, total={len(passed_files)}"
        )
        return 0
    if args.batch_inbox:
        factor_files = sorted(inbox_dir.glob("*.py"))
        if not factor_files:
            print(f"No factor files found in {inbox_dir}")
            return 0
    else:
        factor_files = [Path(args.factor)]

    summary = {"passed": 0, "rejected": 0, "reports": []}

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
        )

        summary[result["status"]] += 1
        summary["reports"].append(result["report_file"])

        reason_preview = "; ".join(result.get("reason", [])[:2])
        print(
            f"[{result['status'].upper()}] {factor_file.name} -> {result.get('moved_to', 'N/A')}"
        )
        if reason_preview:
            print(f"  reason: {reason_preview}")

    print(
        "Summary: "
        f"passed={summary['passed']}, rejected={summary['rejected']}, reports={len(summary['reports'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
