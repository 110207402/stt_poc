# Phase 3 Breeze-ASR-26 4-way Ablation Evaluation

**測試集**：90 條 query × 4 組 ablation = 360 次推論

**模型**：MediaTek-Research/Breeze-ASR-26 (Whisper-large-v2 fine-tune)

**環境**：Colab A100 / BF16 / num_beams=1 / chunk_length_s=30



**4 組 ablation**：

- **E1** baseline

- **E2** +保險詞



**指標說明**：

- `CER` = 嚴格字元錯誤率（去標點/空白後計算 Levenshtein）

- `CER_norm` = 加上 中文數字↔阿拉伯數字 的日期標準化 後計算 (e.g. 民國七十年 ≡ 民國70年)

- `PII recall` = 該欄位個資在 hypothesis 中是否能被找到（normalized 模式：經過去標點/全形半形/中阿數字統一）

- `Domain term recall` = 保險術語在 hypothesis 中是否出現

---

## 1. 總體 CER

| Ablation | CER (strict) | CER (norm) | N |
|---|---|---|---|
| E1 baseline | 16.41% | 16.34% | 90 |
| E2 +保險詞 | 13.51% | 13.54% | 90 |

→ **最佳組合：E2 +保險詞**，CER_norm = 13.54%



## 2. 各語言條件下 CER (CER_norm)

| language | N | E1 | E2 |
|---|---|---|---|
| mandarin | 30 |  5.24% |  2.98% |
| hokkien | 30 | 20.80% | 19.14% |
| codeswitch | 30 | 22.98% | 18.51% |

**觀察**：

- **mandarin** (30 條)：最佳 E2= 2.98%, 最差 E1= 5.24%, 差距  2.26%

- **hokkien** (30 條)：最佳 E2=19.14%, 最差 E1=20.80%, 差距  1.66%

- **codeswitch** (30 條)：最佳 E2=18.51%, 最差 E1=22.98%, 差距  4.46%



## 3. 各 subtype CER (CER_norm)

| subtype | N | E1 | E2 |
|---|---|---|---|
| A1 | 18 | 13.28% | 11.73% |
| A2 | 6 | 15.04% | 12.71% |
| B1 | 15 | 17.03% | 13.64% |
| B2 | 7 | 14.98% | 12.28% |
| C1 | 8 | 14.77% | 10.82% |
| C2 | 6 |  9.26% |  9.31% |
| D | 30 | 20.24% | 16.61% |

**Subtype 對照**：A1=姓名+身分證+生日；A2=A1+保單號；B1=僅姓名；B2=姓名+保單號；

C1=姓名+地址；C2=姓名+電話；D=無 PII



## 4. PII 各欄位召回率（normalized mode）

| PII 類型 | N (出現次數) | E1 | E2 |
|---|---|---|---|
| name | 60 | 76.67% | 76.67% |
| national_id | 24 | 45.83% | 50.00% |
| birth_date | 24 | 100.00% | 100.00% |
| policy_id | 13 | 61.54% | 69.23% |
| phone | 6 | 66.67% | 66.67% |
| address | 8 | 37.50% | 50.00% |



**3 種匹配模式對比**（headline 數字為 normalized）：

### 4.1 name (N=60)

| Ablation | strict | normalized | fuzzy |
|---|---|---|---|
| E1 baseline | 76.67% | 76.67% | 93.33% |
| E2 +保險詞 | 76.67% | 76.67% | 96.67% |

### 4.2 national_id (N=24)

| Ablation | strict | normalized | fuzzy |
|---|---|---|---|
| E1 baseline | 33.33% | 45.83% | 87.50% |
| E2 +保險詞 | 41.67% | 50.00% | 83.33% |

### 4.3 birth_date (N=24)

| Ablation | strict | normalized | fuzzy |
|---|---|---|---|
| E1 baseline | 66.67% | 100.00% | 100.00% |
| E2 +保險詞 | 79.17% | 100.00% | 100.00% |

### 4.4 policy_id (N=13)

| Ablation | strict | normalized | fuzzy |
|---|---|---|---|
| E1 baseline | 15.38% | 61.54% | 69.23% |
| E2 +保險詞 | 15.38% | 69.23% | 69.23% |

### 4.5 phone (N=6)

| Ablation | strict | normalized | fuzzy |
|---|---|---|---|
| E1 baseline |  0.00% | 66.67% | 100.00% |
| E2 +保險詞 |  0.00% | 66.67% | 100.00% |

### 4.6 address (N=8)

| Ablation | strict | normalized | fuzzy |
|---|---|---|---|
| E1 baseline |  0.00% | 37.50% | 87.50% |
| E2 +保險詞 | 12.50% | 50.00% | 100.00% |



## 5. PII recall × language（normalized）

| PII 類型 | language | N | E1 | E2 |
|---|---|---|---|---|
| name | mandarin | 20 | 95.00% | 95.00% |
| name | hokkien | 20 | 45.00% | 45.00% |
| name | codeswitch | 20 | 90.00% | 90.00% |
| national_id | mandarin | 8 | 50.00% | 50.00% |
| national_id | hokkien | 8 | 25.00% | 37.50% |
| national_id | codeswitch | 8 | 62.50% | 62.50% |
| birth_date | mandarin | 8 | 100.00% | 100.00% |
| birth_date | hokkien | 8 | 100.00% | 100.00% |
| birth_date | codeswitch | 8 | 100.00% | 100.00% |
| policy_id | mandarin | 4 | 75.00% | 100.00% |
| policy_id | hokkien | 5 | 20.00% | 20.00% |
| policy_id | codeswitch | 4 | 100.00% | 100.00% |
| phone | mandarin | 2 | 100.00% | 100.00% |
| phone | hokkien | 2 | 50.00% | 50.00% |
| phone | codeswitch | 2 | 50.00% | 50.00% |
| address | mandarin | 3 | 33.33% | 66.67% |
| address | hokkien | 2 | 50.00% | 50.00% |
| address | codeswitch | 3 | 33.33% | 33.33% |



## 6. 保險術語召回率

| Ablation | strict | normalized | fuzzy | 命中數 |
|---|---|---|---|---|
| E1 baseline | 33.68% | 34.21% | 62.63% | 65 |
| E2 +保險詞 | 58.42% | 60.53% | 81.58% | 115 |

**N (術語實例) = 190，獨立術語數 = 58**



### 6.1 各術語在不同 ablation 的命中數（normalized）

| 術語 | 出現次數 | E1 | E2 |
|---|---|---|---|
| 凱基 | 41 | 14 | 28 |
| 享安心 | 11 | 0 | 2 |
| 新康泰 | 10 | 4 | 7 |
| 終身壽險 | 9 | 1 | 9 |
| 心康泰 | 8 | 0 | 2 |
| 長照險 | 8 | 0 | 2 |
| 投資型保單 | 7 | 5 | 7 |
| 增優利 | 7 | 0 | 0 |
| 住院醫療附約 | 6 | 1 | 5 |
| 享放心 | 5 | 0 | 0 |
| 醫療附約 | 5 | 1 | 5 |
| 防癌險 | 5 | 1 | 1 |
| 醫療險 | 4 | 4 | 4 |
| 重大疾病險 | 4 | 1 | 1 |
| 投資型壽險 | 3 | 0 | 3 |
| 等待期 | 3 | 3 | 3 |
| 年金保險 | 3 | 2 | 1 |
| 認定失能 | 3 | 1 | 2 |
| 長照保險 | 2 | 0 | 1 |
| 住院理賠 | 2 | 1 | 2 |
| 年金給付 | 2 | 1 | 2 |
| 儲蓄險 | 2 | 1 | 1 |
| 傷害險 | 2 | 2 | 2 |
| 解約金 | 2 | 2 | 2 |
| 年金 | 2 | 2 | 2 |
| 長照保單 | 2 | 0 | 0 |

（僅顯示出現 ≥2 次的術語）



## 7. 最差 case（CER_norm 最高的 5 條，每組 ablation）

### E1 baseline

| case_id | language | subtype | CER_norm | REF (前 30 字) |
|---|---|---|---|---|
| q022 | hokkien | A2 | 52.05% | 你好我是黃美玲，身分證G134567890，民國六十年四月一… |
| q082 | codeswitch | D | 43.59% | 想請問凱基心康泰跟新康泰這兩張住院醫療附約有什麼差別，要不要… |
| q088 | codeswitch | D | 43.24% | 想請教重大疾病險的保險金跟癌症險的給付可以同時申請嗎，會不會… |
| q079 | hokkien | D | 41.67% | 凱基享安心這張保單，停繳之後多久會失效，可以申請復效嗎，復效… |
| q089 | codeswitch | D | 41.46% | 請問凱基新康泰住院醫療附約有沒有限制總給付次數或天數，慢性病… |

### E2 +保險詞

| case_id | language | subtype | CER_norm | REF (前 30 字) |
|---|---|---|---|---|
| q022 | hokkien | A2 | 47.95% | 你好我是黃美玲，身分證G134567890，民國六十年四月一… |
| q008 | hokkien | A1 | 38.60% | 你好我許文傑，身分證H251234567，民國七十五年六月十… |
| q085 | codeswitch | D | 35.00% | 您好，凱基增優利這張年金保險的預定利率是按哪一年的標準計算，… |
| q071 | hokkien | D | 33.33% | 請問凱基享放心這張長照保險，要怎樣才符合認定失能的標準，神經… |
| q088 | codeswitch | D | 32.43% | 想請教重大疾病險的保險金跟癌症險的給付可以同時申請嗎，會不會… |

## 8. E2 相對 E1 的最大改善 / 退步

### E2 +保險詞 vs E1 baseline

#### Top 5 改善（E1 較差，此 ablation 救回）

| case_id | lang | subtype | E1 CER | E2 CER | 改善幅度 |
|---|---|---|---|---|---|
| q084 | codeswitch | D | 32.43% | 10.81% | 21.62% |
| q082 | codeswitch | D | 43.59% | 23.08% | 20.51% |
| q049 | codeswitch | B1 | 29.73% | 16.22% | 13.51% |
| q066 | mandarin | D | 12.50% |  0.00% | 12.50% |
| q053 | codeswitch | B1 | 36.11% | 25.00% | 11.11% |

#### Top 5 退步（此 ablation 反而變差）

| case_id | lang | subtype | E1 CER | E2 CER | 退步幅度 |
|---|---|---|---|---|---|
| q071 | hokkien | D | 15.38% | 33.33% | 17.95% |
| q014 | codeswitch | A1 | 12.28% | 26.32% | 14.04% |
| q075 | hokkien | D | 25.00% | 30.56% |  5.56% |
| q085 | codeswitch | D | 30.00% | 35.00% |  5.00% |
| q080 | hokkien | D | 29.41% | 32.35% |  2.94% |

## 9. 結論與建議

**最佳整體配置：E2 +保險詞**，CER_norm 從 baseline 的 16.34% 降至 13.54%，相對改善 17.1%。



**Prompt 干預效果**：

- **E2 +保險詞** 相對 E1 絕對改善  2.79% (相對 17.1%)



**PII 召回觀察**：

- **name** (N=60): E1=76.67%, 最佳 E1=76.67% (持平)

- **national_id** (N=24): E1=45.83%, 最佳 E2=50.00% (+ 4.17%)

- **birth_date** (N=24): E1=100.00%, 最佳 E1=100.00% (持平)

- **policy_id** (N=13): E1=61.54%, 最佳 E2=69.23% (+ 7.69%)

- **phone** (N=6): E1=66.67%, 最佳 E1=66.67% (持平)

- **address** (N=8): E1=37.50%, 最佳 E2=50.00% (+12.50%)



---

_evaluation script: `benchmark/breeze_asr26/scripts/eval_asr_results.py`_
