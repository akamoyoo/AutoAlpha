# Factor Evaluation Report: momximb_60_v1

## 1. 数据与评估设置
- 数据文件: `Data/BTCUSDT/klines_1s_monthly_filled_all_with_logret_1h.parquet`
- 目标列: `log_return`
- 评估脚本: `factors/evaluate_factor.py`
- 评估命令:
  - `python factors/evaluate_factor.py --factor factors/inbox/momximb_60_v1.py`

## 2. 新因子定义
- 因子文件: `factors/passed/momximb_60_v1.py`
- 因子名: `momximb_60_v1`
- 核心思想:
  - `60s` 对数动量: `sum(diff(log(close)), window=60)`
  - `60s` 主动买盘失衡均值: `mean((2*taker_buy_base_asset_volume - volume)/volume, window=60)`
  - 因子值: 两者乘积
- 使用列:
  - `close`
  - `volume`
  - `taker_buy_base_asset_volume`

## 3. 评估结果
- 结论: **Passed**
- 文件流转:
  - `factors/inbox/momximb_60_v1.py` -> `factors/passed/momximb_60_v1.py`
- JSON 报告:
  - `factors/reports/momximb_60_v1_20260409_000031.json`

### 关键指标
- coverage: `0.999125122928187`
- unique_ratio: `0.9678729927610316`
- std: `0.00014888397714149894`
- pearson: `0.006729230147784916`
- spearman: `0.0041156000624221435`
- quantile_spread: `6.538540114860495e-05`
- valid_count: `102380747`
- total_count: `102470396`

### 阈值对照（默认）
- coverage >= `0.8` -> 通过
- unique_ratio >= `0.001` -> 通过
- std >= `1e-8` -> 通过
- |pearson| >= `0.005` 或 |spearman| >= `0.005` -> 通过（由 Pearson 通过）
- 与已通过因子相关性 |corr| <= `0.9` -> 通过（当前无已通过因子参与对比）

## 4. 简要解读
- 因子具备高覆盖、较高离散度与稳定可计算性。
- 预测性门槛通过主要来自 Pearson 相关性（线性相关略高于阈值）。
- Spearman 略低于阈值，但规则为 Pearson/Spearman 二选一通过，因此整体判定通过。
