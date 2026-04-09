# Factor Evaluation Report: toxicrev_3600_v1

## 1. 数据与评估设置
- 数据文件: `Data/BTCUSDT/klines_1s_monthly_filled_all_with_logret_1h.parquet`
- 目标列: `log_return`（一小时级别）
- 评估脚本: `factors/evaluate_factor.py`
- 评估命令:
  - `python factors/evaluate_factor.py --factor factors/inbox/toxicrev_3600_v1.py`

## 2. 新因子定义
- 因子文件: `factors/passed/toxicrev_3600_v1.py`
- 因子名: `toxicrev_3600_v1`
- 核心思想:
  - `ret_1s = diff(log(close))`
  - 报价口径主动买盘失衡: `quote_buy_imbalance = (2*taker_buy_quote_asset_volume - quote_asset_volume)/quote_asset_volume`
  - `3600s` 订单流毒性: `toxicity_3600 = mean(abs(quote_buy_imbalance), window=3600)`
  - `3600s` 反转动量: `reversal_3600 = -sum(ret_1s, window=3600)`
  - 因子值: `factor = toxicity_3600 * reversal_3600`
- 使用列:
  - `close`
  - `quote_asset_volume`
  - `taker_buy_quote_asset_volume`

## 3. 评估结果
- 结论: **Passed**
- 文件流转:
  - `factors/inbox/toxicrev_3600_v1.py` -> `factors/passed/toxicrev_3600_v1.py`
- JSON 报告:
  - `factors/reports/toxicrev_3600_v1_20260409_004104.json`

### 关键指标
- coverage: `0.9991558244783205`
- unique_ratio: `0.9811147540560896`
- std: `0.003074331906443399`
- pearson: `0.013570160647457538`
- spearman: `0.046794890139578736`
- quantile_spread: `0.00014529342947763293`
- valid_count: `102383893`
- total_count: `102470396`

### 阈值对照（默认）
- coverage >= `0.8` -> 通过
- unique_ratio >= `0.001` -> 通过
- std >= `1e-8` -> 通过
- |pearson| >= `0.005` 或 |spearman| >= `0.005` -> 通过（Pearson 与 Spearman 均通过）
- 与已通过因子相关性 |corr| <= `0.9` -> 通过
  - 与 `flowvol_120_v1` 的 |corr| = `0.02500964594775827`
  - 与 `momximb_60_v1` 的 |corr| = `0.026397924428961817`
  - 与 `revimb_300_v1` 的 |corr| = `0.018902361271943464`
  - 与 `toxicrev_300_v1` 的 |corr| = `0.28480694014555097`

## 4. 简要解读
- 因子覆盖率和唯一值比例都很高，工程可计算性稳定。
- 针对一小时 `log_return`，线性相关与秩相关均显著超过默认阈值，预测性较强。
- 与现有通过因子相关性均远低于 0.9，共线性风险低，可加入候选池。
