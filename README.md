# 台股方向預測實驗

本專案以漸進方式驗證不同資料群是否能改善五日方向預測，不一次加入新聞、LLM 或深度學習模型。

## 版本

| 版本 | 累積使用的資料 |
| --- | --- |
| V1 | 價格與技術面 |
| V2 | V1 + 籌碼面 |
| V3 | V2 + 基本面 |
| V4 | V3 + 市場環境 |
| V5（資料層） | V4 + 新聞活動特徵；尚未加入模型選拔 |

每個版本都使用相同的日期切分、XGBoost 參數與評估指標，方便比較新增資料是否真的有效。

## V5 新聞資料層

V5 先完成資料收集與 point-in-time 特徵，不變更目前的 V2 champion，也不使用
LLM 或情緒模型。FinMind `TaiwanStockNews` 單次只回傳一天，因此程式會逐日下載並
快取至 `data/raw/news/<股票代號>/`。預設每次最多請求 31 個尚未快取的日期，避免
誤觸大量 API 請求。

```powershell
.\.venv\Scripts\python.exe build_news_features.py `
  --stock-id 2330 `
  --start-date 2026-08-01 `
  --end-date 2026-08-24
```

若需要 FinMind token，可先設定 `FINMIND_TOKEN` 環境變數。加上
`--merge-dataset` 才會另外建立 `2330_dataset_v5_ready.csv`；原本的
`2330_dataset.csv`、V2 模型與 `models/registry.json` 都不會被覆寫：

```powershell
.\.venv\Scripts\python.exe build_news_features.py `
  --stock-id 2330 `
  --start-date 2026-08-01 `
  --end-date 2026-08-24 `
  --merge-dataset
```

新聞會先依正規化網址去重與清除 HTML。發布時間不晚於台股收盤時間 `13:30`
的新聞可進入當日特徵；收盤後、週末或休市日新聞延至下一交易日，避免時間洩漏。
目前只產生新聞數量、來源數、標題／摘要長度與 3／5 個交易日新聞量等中性活動
特徵，尚未對文字指定正負情緒。輸出只涵蓋實際請求的新聞期間（並保留最後一天
收盤後新聞所需的下一交易日）；合併後區間外維持缺值，不會被誤標成零篇新聞。
若 FinMind 即時回傳未包含舊版文件中的 `description`，管線會保留空摘要並繼續處理，
不會影響標題、來源、網址與發布時間特徵。
3／5 日累積量在尚未具有完整 3／5 個交易日觀察時也維持缺值，避免把不完整回看
窗口誤當成完整新聞量。

- `data/raw/<股票代號>_news_normalized.csv`：去重後新聞與可用日期
- `data/processed/<股票代號>_news_features.csv`：每日新聞活動特徵
- `data/processed/<股票代號>_news_data_quality.json`：涵蓋區間與去重品質摘要
- `data/processed/<股票代號>_dataset_v5_ready.csv`：選擇性合併產物

### 可續傳歷史回補

`backfill_news.py` 會從最新缺口往前下載，預設每次執行一批 31 天。每天各自快取，
所以中途限流或斷線後重跑不會重抓已完成日期；輸出只使用直到目標結束日都沒有
缺口的連續區間。

```powershell
.\.venv\Scripts\python.exe backfill_news.py --stock-id 2330
```

若要讓單次程序持續至預設目標 `2020-04-01`，可將批次上限設為 `0`。長時間執行
仍可能受 FinMind 流量限制，但再次執行會從進度接續：

```powershell
.\.venv\Scripts\python.exe backfill_news.py --stock-id 2330 --max-chunks 0
```

若不熟悉環境變數，可將 token 單獨貼到 `data/private/finmind_token.txt` 後儲存。
回補器會優先使用環境變數，否則讀取這個一次性檔案，並在讀取後立即刪除。
完整回補預設以 token 每 6.6 秒、無 token 每 12.5 秒最多一個請求，保留在官方
每小時額度以下；可用 `--request-delay-seconds` 明確覆寫。

- `data/processed/<股票代號>_news_backfill_progress.json`：回補狀態、連續涵蓋起點與下一個缺口

### V5 新聞品質審計與特徵凍結

完整回補後先執行資料審計，不直接訓練模型：

```powershell
.\.venv\Scripts\python.exe audit_news.py
```

審計會驗證快取無缺日、新聞 ID 唯一、每日加總一致、收盤後新聞確實延至下一
交易日，並輸出年度覆蓋、來源集中度、異常新聞量、連續零新聞與相對 2021 年的
PSI 漂移。摘要欄位若持續缺失會從凍結特徵中排除，避免將常數欄位加入模型。

- `reports/news_audit_summary.json`：硬性檢查、警告與是否可進入 V5 比較
- `reports/news_audit_yearly.csv`：年度新聞量與缺值摘要
- `reports/news_audit_sources.csv`：來源數量、占比與活躍期間
- `reports/news_audit_anomalies.csv`：年度內異常新聞量日期
- `reports/news_audit_zero_streaks.csv`：連續零新聞交易日
- `reports/news_feature_drift.csv`：各特徵年度 PSI
- `reports/news_feature_schema_v5.json`：模型比較前凍結的特徵與 SHA-256

## 執行

預設只訓練 V1 baseline：

```powershell
.\.venv\Scripts\python.exe train.py
```

訓練指定版本：

```powershell
.\.venv\Scripts\python.exe train.py --version v2
```

當 V1 驗證完成後，可一次比較 V1～V4：

```powershell
.\.venv\Scripts\python.exe train.py --version all
```

比較目前已完成的 V1～V3：

```powershell
.\.venv\Scripts\python.exe compare_versions.py
.\.venv\Scripts\python.exe train.py --through v3
```

預設會使用平衡樣本權重，降低模型只預測單一類別的情況。如需無權重對照組：

```powershell
.\.venv\Scripts\python.exe train.py --version v1 --class-weighting none
```

V1 的五日漲跌門檻預設為 `±1.5%`。這個設定是以 2021～2024
四段 walk-forward 驗證比較 `1%`、`1.5%`、`2%`、`2.5%` 與 `3%`
後選出，2025 年以後的測試集完全不參與挑選。
重新執行門檻診斷：

```powershell
.\.venv\Scripts\python.exe diagnose_v1.py
```

也可以明確指定門檻：

```powershell
.\.venv\Scripts\python.exe train.py --version v1 --label-threshold 0.015
```

## 輸出

- `models/xgboost_v1.json`：分版本模型
- `reports/xgboost_v1_metrics.json`：完整分類指標
- `reports/xgboost_v1_feature_importance.csv`：特徵重要度
- `data/processed/2330_predictions_v1.csv`：逐日預測
- `reports/xgboost_version_comparison.csv`：版本比較表
- `reports/xgboost_v1_threshold_diagnostics.csv`：V1 驗證集門檻診斷
- `reports/xgboost_v1_walk_forward_folds.csv`：各 walk-forward 區間結果
- `reports/xgboost_walk_forward_version_comparison.csv`：版本驗證比較
- `reports/xgboost_walk_forward_version_folds.csv`：版本各年度明細

## V1～V3 交易回測

回測採 long-or-cash：預測上漲時持有 2330，預測盤整或下跌時持有現金。
訊號於收盤後產生，下一交易日開盤執行，每五個交易日調整一次，因此持有區間不重疊。

先建立 walk-forward 預測，再執行回測：

```powershell
.\.venv\Scripts\python.exe compare_versions.py
.\.venv\Scripts\python.exe backtest.py
```

預設每次買進或賣出收取 `14.25 bps` 手續費，賣出另收 `30 bps`
股票交易稅。券商手續費可能有折扣，也可能有最低收費，應以實際帳戶條件覆寫：

```powershell
.\.venv\Scripts\python.exe backtest.py --commission-bps 5 --sell-tax-bps 30
```

回測輸出：

- `reports/backtest_version_metrics.csv`：各版本與買進持有績效
- `reports/backtest_version_metrics.json`：相同績效的 JSON 版本
- `reports/backtest_version_trades.csv`：每個五日持有區間
- `reports/backtest_version_yearly_returns.csv`：逐年毛額與淨額報酬
- `data/processed/backtest_version_equity.csv`：毛額與淨額權益曲線

此回測未模擬滑價、最低手續費、成交量限制、股利與公司行動，不能視為實際可成交績效。

## V3 基本面消融

完整 V3 未通過保留測試後，可用下列指令比較 V2 分別加入單一基本面特徵、營收組、估值組與完整基本面：

```powershell
.\.venv\Scripts\python.exe diagnose_v3.py
```

診斷只使用 2021～2024 walk-forward 區間，不再利用 2025 年後測試集挑選特徵。輸出：

- `reports/xgboost_v3_fundamental_ablation.csv`：候選組合摘要
- `reports/xgboost_v3_fundamental_ablation_folds.csv`：逐年結果

## 目前模型狀態

- Champion：V2（技術面 + 籌碼面）
- Candidate：無
- Rejected：完整 V3（V2 + 全部基本面）
- Rejected：V3-lite（V2 + PB），未通過五檔股票外部驗證

建立 V3-lite 候選：

```powershell
.\.venv\Scripts\python.exe train_v3_lite.py
```

此模型只使用 2025 年以前、完成五日邊界 purge 的資料訓練，保留為失敗實驗的
可重現產物，不應用於正式預測。

- `models/registry.json`：Champion、Candidate 與 Rejected 狀態
- `models/xgboost_v3_lite_candidate.json`：候選模型
- `reports/xgboost_v3_lite_candidate.json`：候選訓練資料與限制

## 多股票外部驗證

外部股票池在下載結果前固定為 `2317、2454、2303、2881、1301`。每檔分別使用
相同標籤、V2／V3-lite 特徵、walk-forward 切分與交易成本：

```powershell
.\.venv\Scripts\python.exe external_validate.py
```

V3-lite 在五檔外部測試中，Balanced accuracy 與 Macro F1 均只在 `1/5` 勝過 V2，
因此 PB 的 2330 改善未能跨股票重現，模型登錄狀態已改為 rejected。

- `reports/external_validation_classification.csv`：逐股票分類結果
- `reports/external_validation_classification_summary.csv`：分類摘要
- `reports/external_validation_backtest.csv`：逐股票回測結果
- `reports/external_validation_backtest_summary.csv`：回測摘要
- `reports/external_validation_decision.json`：升版決策

## Pooled V2 實驗

六檔股票合併訓練的原始 V2 與尺度化 V2 都已測試：

```powershell
.\.venv\Scripts\python.exe train_pooled_v2.py
.\.venv\Scripts\python.exe train_pooled_v2_scaled.py
```

尺度化版本將 MACD 系列除以股價、法人與融資券流量除以同期成交量、餘額除以
20 日平均成交量。即使修正跨股票尺度，外部測試的 Balanced accuracy 與 Macro F1
仍都只在 2/6 股票勝過逐股票 V2，因此兩個 pooled 版本均標記為 rejected。

- `reports/pooled_v2_decision.json`：原始 pooled 決策
- `reports/pooled_v2_scaled_decision.json`：尺度化 pooled 決策
- `reports/pooled_v2_vs_independent_classification.csv`：原始 pooled 分類比較
- `reports/pooled_v2_scaled_vs_independent_classification.csv`：尺度化分類比較
- `reports/pooled_v2_vs_independent_backtest.csv`：原始 pooled 回測比較
- `reports/pooled_v2_scaled_vs_independent_backtest.csv`：尺度化回測比較

## 五日報酬回歸與排序

將三分類目標改為連續五日報酬，並每天對六檔股票做橫斷面排名：

```powershell
.\.venv\Scripts\python.exe train_return_ranker.py
```

四個 walk-forward fold 的平均 rank IC 都為正，但每五日切換至預測第一名造成高換手；
walk-forward top-1 毛報酬為正，扣除成本後轉負。因此此版本登記為 research signal，
不是 champion 或可交易 candidate。

- `reports/return_ranker_metrics.csv`：MAE、RMSE 與 rank IC
- `reports/return_ranker_backtest.csv`：top-1 與六檔等權比較
- `reports/return_ranker_intervals.csv`：每個五日排名持有區間
- `reports/return_ranker_decision.json`：研究決策

## 低換手排序策略

只用 2021～2024 walk-forward 預測比較排名緩衝、換股優勢與最低預測報酬：

```powershell
.\.venv\Scripts\python.exe optimize_rank_strategy.py
```

預先設定的 gate 為換手至少減半、淨報酬為正、四年至少三年獲利，且最大回撤
相對原 top-1 策略不得再惡化超過 5 個百分點。36 組規則中只有 2 組通過；
選定規則為「目前持股仍在前 3 名且預測報酬至少 0.25% 就續抱」。此規則只登記為
全期間候選，不能直接登記為 shadow candidate。

- `reports/rank_strategy_grid.csv`：36 組完整結果
- `reports/rank_strategy_yearly.csv`：逐年結果
- `reports/rank_strategy_decision.json`：gate 與選定規則
- `data/processed/rank_strategy_selected_intervals.csv`：選定規則持有區間

策略層再使用 nested walk-forward 驗證：2021 選 2022、2021～2022 選 2023、
2021～2023 選 2024。

```powershell
.\.venv\Scripts\python.exe validate_rank_strategy_nested.py
```

2021 選出的規則在 2022 虧損 38.49%；之後沒有規則能通過 gate，因此 2023、2024
持有現金。低換手策略已撤銷，不再是 shadow candidate。

- `reports/rank_strategy_nested_selections.csv`：每年只用過去資料的選擇
- `reports/rank_strategy_nested_metrics.json`：nested 總結果
- `data/processed/rank_strategy_nested_intervals.csv`：nested 持有區間

目前輸出只用於研究與資料管線驗證，不代表可直接用於實際交易。
