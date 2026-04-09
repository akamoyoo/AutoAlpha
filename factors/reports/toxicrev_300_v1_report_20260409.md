# Factor Evaluation Report: toxicrev_300_v1

## 1. 数据与评估设置
- 数据文件: `Data/BTCUSDT/klines_1s_monthly_filled_all_with_logret_1h.parquet`
- 目标列: `log_return`（一小时级别）
- 评估脚本: `factors/evaluate_factor.py`
- 评估命令:
  - `python factors/evaluate_factor.py --factor factors/inbox/toxicrev_300_v1.py`

## 2. 新因子定义
- 因子文件: `factors/passed/toxicrev_300_v1.py`
- 因子名: `toxicrev_300_v1`
- 核心思想:
  - `ret_1s = diff(log(close))`
  - 报价口径主动买盘失衡: `quote_buy_imbalance = (2*taker_buy_quote_asset_volume - quote_asset_volume)/quote_asset_volume`
  - `300s` 订单流毒性: `toxicity_300 = mean(abs(quote_buy_imbalance), window=300)`
  - `300s` 反转动量: `reversal_300 = -sum(ret_1s, window=300)`
  - 因子值: `factor = toxicity_300 * reversal_300`
- 使用列:
  - `close`
  - `quote_asset_volume`
  - `taker_buy_quote_asset_volume`

## 3. 评估结果
- 结论: **Passed**
- 文件流转:
  - `factors/inbox/toxicrev_300_v1.py` -> `factors/passed/toxicrev_300_v1.py`
- JSON 报告:
  - `factors/reports/toxicrev_300_v1_20260409_003022.json`

### 关键指标
- coverage: `0.9991274650680573`
- unique_ratio: `0.9773564499822609`
- std: `0.0008943394034618434`
- pearson: `0.01024331459173326`
- spearman: `0.027223178396113417`
- quantile_spread: `0.00011636334756261996`
- valid_count: `102380987`
- total_count: `102470396`

### 阈值对照（默认）
- coverage >= `0.8` -> 通过
- unique_ratio >= `0.001` -> 通过
- std >= `1e-8` -> 通过
- |pearson| >= `0.005` 或 |spearman| >= `0.005` -> 通过（Pearson 与 Spearman 均通过）
- 与已通过因子相关性 |corr| <= `0.9` -> 通过
  - 与 `flowvol_120_v1` 的 |corr| = `0.01017551964523845`
  - 与 `momximb_60_v1` 的 |corr| = `0.014031379642714604`
  - 与 `revimb_300_v1` 的 |corr| = `0.012310283374772786`

## 4. 简要解读
- 因子覆盖率高、唯一值比例高、波动稳定，工程可用性良好。
- 对一小时 `log_return` 的线性与秩相关均超过门槛，预测性优于当前门槛要求。
- 与现有通过因子共线性很低，可作为相对独立的新信号加入候选池。
