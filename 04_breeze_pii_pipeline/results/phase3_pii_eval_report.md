# Phase 3 PII Detection Evaluation — M1 vs M2

**測試集**：180 (case × ablation) × 2 methods = 360 次偵測

**ASR 上游**：Breeze-ASR-26 在 E1 baseline + E2 +保險詞 兩組設定的輸出

**PII 方法**：

- **M1** OpenAI Privacy Filter (HF: `openai/privacy-filter`) — 8 native categories 映射成我們 6 類

- **M2** Azure GPT-4o-mini — JSON schema 強制輸出 6 類

**比對標準**（容許 ASR 帶來的字元錯誤）：

- **exact**：偵測到的字串完全包含 GT 值

- **normalized**：去標點/全形半形/中阿數字統一後包含

- **fuzzy**：normalized 後 Levenshtein ≤ 類別 threshold (name=1, id=1, date=2, addr=4 ...)

- **no**：沒命中

---

## 1. 整體 micro-recall（所有 PII instance 池在一起）

| Method | Ablation | PII 總數 | 命中 | Recall |
|---|---|---|---|---|
| M1 OpenAI Privacy Filter | E1 baseline | 135 | 63 | 46.67% |
| M1 OpenAI Privacy Filter | E2 +保險詞 | 135 | 65 | 48.15% |
| M2 Azure GPT-4o-mini | E1 baseline | 135 | 128 | 94.81% |
| M2 Azure GPT-4o-mini | E2 +保險詞 | 135 | 127 | 94.07% |

## 2. 各 PII 類別 recall（method × ablation）

| PII 類型 | N | M1/E1 | M1/E2 | M2/E1 | M2/E2 |
|---|---|---|---|---|---|
| name | 60 | 65.00% | 68.33% | 93.33% | 96.67% |
| national_id | 24 |  8.33% |  8.33% | 95.83% | 83.33% |
| birth_date | 24 | 29.17% | 29.17% | 100.00% | 100.00% |
| policy_id | 13 | 30.77% | 38.46% | 92.31% | 92.31% |
| phone | 6 | 66.67% | 66.67% | 83.33% | 83.33% |
| address | 8 | 87.50% | 75.00% | 100.00% | 100.00% |

## 3. M1 vs M2 同 ablation 直接對比

### E1 baseline

| PII 類型 | N | M1 recall | M2 recall | M2-M1 差距 |
|---|---|---|---|---|
| name | 60 | 65.00% | 93.33% | +28.33% |
| national_id | 24 |  8.33% | 95.83% | +87.50% |
| birth_date | 24 | 29.17% | 100.00% | +70.83% |
| policy_id | 13 | 30.77% | 92.31% | +61.54% |
| phone | 6 | 66.67% | 83.33% | +16.67% |
| address | 8 | 87.50% | 100.00% | +12.50% |

### E2 +保險詞

| PII 類型 | N | M1 recall | M2 recall | M2-M1 差距 |
|---|---|---|---|---|
| name | 60 | 68.33% | 96.67% | +28.33% |
| national_id | 24 |  8.33% | 83.33% | +75.00% |
| birth_date | 24 | 29.17% | 100.00% | +70.83% |
| policy_id | 13 | 38.46% | 92.31% | +53.85% |
| phone | 6 | 66.67% | 83.33% | +16.67% |
| address | 8 | 75.00% | 100.00% | +25.00% |

## 4. Recall × language（normalized / micro per language-PII pair）

### E1 baseline

| PII 類型 | language | N | M1 | M2 |
|---|---|---|---|---|
| name | codeswitch | 20 | 60.00% | 100.00% |
| name | hokkien | 20 | 50.00% | 85.00% |
| name | mandarin | 20 | 85.00% | 95.00% |
| national_id | codeswitch | 8 |  0.00% | 100.00% |
| national_id | hokkien | 8 | 25.00% | 87.50% |
| national_id | mandarin | 8 |  0.00% | 100.00% |
| birth_date | codeswitch | 8 | 50.00% | 100.00% |
| birth_date | hokkien | 8 |  0.00% | 100.00% |
| birth_date | mandarin | 8 | 37.50% | 100.00% |
| policy_id | codeswitch | 4 |  0.00% | 100.00% |
| policy_id | hokkien | 5 | 40.00% | 80.00% |
| policy_id | mandarin | 4 | 50.00% | 100.00% |
| phone | codeswitch | 2 | 50.00% | 50.00% |
| phone | hokkien | 2 | 100.00% | 100.00% |
| phone | mandarin | 2 | 50.00% | 100.00% |
| address | codeswitch | 3 | 66.67% | 100.00% |
| address | hokkien | 2 | 100.00% | 100.00% |
| address | mandarin | 3 | 100.00% | 100.00% |

### E2 +保險詞

| PII 類型 | language | N | M1 | M2 |
|---|---|---|---|---|
| name | codeswitch | 20 | 70.00% | 100.00% |
| name | hokkien | 20 | 40.00% | 90.00% |
| name | mandarin | 20 | 95.00% | 100.00% |
| national_id | codeswitch | 8 |  0.00% | 87.50% |
| national_id | hokkien | 8 | 25.00% | 75.00% |
| national_id | mandarin | 8 |  0.00% | 87.50% |
| birth_date | codeswitch | 8 | 50.00% | 100.00% |
| birth_date | hokkien | 8 | 12.50% | 100.00% |
| birth_date | mandarin | 8 | 25.00% | 100.00% |
| policy_id | codeswitch | 4 | 25.00% | 100.00% |
| policy_id | hokkien | 5 | 40.00% | 80.00% |
| policy_id | mandarin | 4 | 50.00% | 100.00% |
| phone | codeswitch | 2 | 50.00% | 50.00% |
| phone | hokkien | 2 | 100.00% | 100.00% |
| phone | mandarin | 2 | 50.00% | 100.00% |
| address | codeswitch | 3 | 33.33% | 100.00% |
| address | hokkien | 2 | 100.00% | 100.00% |
| address | mandarin | 3 | 100.00% | 100.00% |

## 5. 假陽性分析（D-subtype 30 條無 PII query）

| Method | Ablation | N (D-cases) | 總 FP 數 | 有 FP 的 case 數 | Case FP rate | 平均 FP/case |
|---|---|---|---|---|---|---|
| M1 OpenAI Privacy Filter | E1 baseline | 30 | 0 | 0 |  0.00% | 0.00 |
| M1 OpenAI Privacy Filter | E2 +保險詞 | 30 | 1 | 1 |  3.33% | 0.03 |
| M2 Azure GPT-4o-mini | E1 baseline | 30 | 3 | 3 | 10.00% | 0.10 |
| M2 Azure GPT-4o-mini | E2 +保險詞 | 30 | 2 | 2 |  6.67% | 0.07 |

**理解**：D-subtype 全部沒有 PII，任何偵測都是假陽性。

`Case FP rate` = 至少有 1 筆假陽性的 case 比例；`平均 FP/case` = 每條 D query 平均偵測幾個假陽。



**FP 類型分布**：

- M1 OpenAI Privacy Filter / E2 +保險詞: name=1

- M2 Azure GPT-4o-mini / E1 baseline: policy_id=3

- M2 Azure GPT-4o-mini / E2 +保險詞: policy_id=2



## 6. 命中模式分布

（在所有有 PII 的 case 中，每個欄位的 recall 落在哪個 match level）

| Method | Ablation | exact | normalized | fuzzy | no (miss) |
|---|---|---|---|---|---|
| M1 OpenAI Privacy Filter | E1 baseline | 39 (28.89%) | 10 ( 7.41%) | 14 (10.37%) | 72 (53.33%) |
| M1 OpenAI Privacy Filter | E2 +保險詞 | 44 (32.59%) | 9 ( 6.67%) | 12 ( 8.89%) | 70 (51.85%) |
| M2 Azure GPT-4o-mini | E1 baseline | 83 (61.48%) | 22 (16.30%) | 23 (17.04%) | 7 ( 5.19%) |
| M2 Azure GPT-4o-mini | E2 +保險詞 | 86 (63.70%) | 20 (14.81%) | 21 (15.56%) | 8 ( 5.93%) |

## 7. M1 / M2 一致性

| Ablation | N (PII 欄位數) | agreement | 兩者皆中 | 兩者皆漏 | 僅 M1 中 | 僅 M2 中 |
|---|---|---|---|---|---|---|
| E1 baseline | 135 | 48.89% | 61 | 5 | 2 | 67 |
| E2 +保險詞 | 135 | 52.59% | 64 | 7 | 1 | 63 |

## 8. 結論

**最佳組合**：M2 Azure GPT-4o-mini 在 E1 baseline 上 micro-recall = 94.81%



- **M1 OpenAI Privacy Filter**：E1=46.67%, E2=48.15%, 差距= 1.48% → E2 prompt 提升 PII recall

- **M2 Azure GPT-4o-mini**：E1=94.81%, E2=94.07%, 差距=-0.74% → E2 prompt 對 PII recall 無顯著影響



**FP 觀察**：

- **M1 OpenAI Privacy Filter**：D queries case FP rate E1= 0.00% / E2= 3.33%

- **M2 Azure GPT-4o-mini**：D queries case FP rate E1=10.00% / E2= 6.67%



_evaluation script: `benchmark/breeze_asr26/scripts/run_pii_eval.py`_

_raw cache: `breeze_asr26/results/phase3_pii_detections.json`_
