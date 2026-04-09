# Factor Evaluation Report: flowvol_120_v1

## 1. 数据与评估设置
- 数据文件: `Data/BTCUSDT/klines_1s_monthly_filled_all_with_logret_1h.parquet`
- 目标列: `log_return`
- 评估脚本: `factors/evaluate_factor.py`
- 评估命令:
  - `python factors/evaluate_factor.py --factor factors/inbox/flowvol_120_v1.py`

## 2. 新因子定义
- 因子文件: `factors/passed/flowvol_120_v1.py`
- 因子名: `flowvol_120_v1`
- 核心思想:
  - `ret_1s = diff(log(close))`
  - 买卖失衡: `(2*taker_buy_base_asset_volume - volume)/volume`
  - `120s` 耦合流: `sum(ret_1s * buy_imbalance, window=120)`
  - `120s` 波动率归一化: `factor = coupled_flow_120 / (std(ret_1s,120) + 1e-12)`
- 使用列:
  - `close`
  - `volume`
  - `taker_buy_base_asset_volume`

## 3. 评估结果
- 结论: **Passed**
- 文件流转:
  - `factors/inbox/flowvol_120_v1.py` -> `factors/passed/flowvol_120_v1.py`
- JSON 报告:
  - `factors/reports/flowvol_120_v1_20260409_000821.json`

### 关键指标
- coverage: `0.9991253864189223`
- unique_ratio: `0.8353766596841707`
- std: `9.091170541254282`
- pearson: `-0.0015708125147519982`
- spearman: `0.009398345922537264`
- quantile_spread: `-5.1227294074461345e-05`
- valid_count: `102380774`
- total_count: `102470396`

### 阈值对照（默认）
- coverage >= `0.8` -> 通过
- unique_ratio >= `0.001` -> 通过
- std >= `1e-8` -> 通过
- |pearson| >= `0.005` 或 |spearman| >= `0.005` -> 通过（由 Spearman 通过）
- 与已通过因子相关性 |corr| <= `0.9` -> 通过
  - 与 `momximb_60_v1` 的 |corr| = `0.20105232456153765`

## 4. 简要解读
- 因子覆盖率高、离散度高，具备稳定可计算性。
- 线性相关（Pearson）较弱，但秩相关（Spearman）超过阈值，说明对目标的单调关系更明显。
- 与当前已通过因子共线性较低，可作为互补信号加入候选池。
