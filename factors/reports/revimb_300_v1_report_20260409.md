# Factor Evaluation Report: revimb_300_v1

## 1. 数据与评估设置
- 数据文件: `Data/BTCUSDT/klines_1s_monthly_filled_all_with_logret_1h.parquet`
- 目标列: `log_return`
- 评估脚本: `factors/evaluate_factor.py`
- 评估命令:
  - `python factors/evaluate_factor.py --factor factors/inbox/revimb_300_v1.py`

## 2. 新因子定义
- 因子文件: `factors/passed/revimb_300_v1.py`
- 因子名: `revimb_300_v1`
- 核心思想:
  - `ret_1s = diff(log(close))`
  - `300s` 动量: `momentum_300 = sum(ret_1s, window=300)`
  - 主动买盘失衡: `buy_imbalance = (2*taker_buy_base_asset_volume - volume)/volume`
  - `300s` 失衡均值: `imbalance_300 = mean(buy_imbalance, window=300)`
  - 因子值: `factor = - momentum_300 * imbalance_300`
- 使用列:
  - `close`
  - `volume`
  - `taker_buy_base_asset_volume`

## 3. 评估结果
- 结论: **Passed**
- 文件流转:
  - `factors/inbox/revimb_300_v1.py` -> `factors/passed/revimb_300_v1.py`
- JSON 报告:
  - `factors/reports/revimb_300_v1_20260409_001634.json`

### 关键指标
- coverage: `0.9991274650680573`
- unique_ratio: `0.9853789551765114`
- std: `0.00019106643750604327`
- pearson: `-0.005230645840540892`
- spearman: `-0.0036271168582368576`
- quantile_spread: `-3.759868012937022e-05`
- valid_count: `102380987`
- total_count: `102470396`

### 阈值对照（默认）
- coverage >= `0.8` -> 通过
- unique_ratio >= `0.001` -> 通过
- std >= `1e-8` -> 通过
- |pearson| >= `0.005` 或 |spearman| >= `0.005` -> 通过（由 Pearson 通过）
- 与已通过因子相关性 |corr| <= `0.9` -> 通过
  - 与 `flowvol_120_v1` 的 |corr| = `0.11033741324453897`
  - 与 `momximb_60_v1` 的 |corr| = `0.2580130263733378`

## 4. 简要解读
- 因子覆盖率高、离散度高，可计算性稳定。
- 预测性通过主要来自 Pearson（绝对值略高于阈值），Spearman 较弱。
- 与现有通过因子共线性较低，可作为补充信号继续纳入候选池。
