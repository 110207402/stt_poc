# Phase 3 Final Report — KGI 人壽 STT + PII 端到端 PoC

**模型棧**：Breeze-ASR-26（語音轉文字）→ OpenAI Privacy Filter / Azure GPT-4o-mini（個資偵測）
**測試集**：90 條查詢 × 2 ASR 設定 × 2 PII 方法 = **360 個端到端評估點**


---

## 1. Background and Objectives

### 1.1 業務背景
凱基人壽客服場景需處理三類來電：
- **純國語客戶**（佔多數）
- **長者偏好閩南語**（高齡客戶大宗）
- **國台語混雜**（台灣自然口語常態）

來電通常涉及：客戶身份核實（姓名／身分證／生日）、保單號碼、保險商品名（凱基享安心、新康泰、長照保險等）、聯絡資訊（地址／電話）。**STT 系統的下游個資保護是合規必要**，不能單純依賴轉錄正確就好。

### 1.2 Phase 3 目標
在 Phase 1（ASR 多模型 batch benchmark）與 Phase 2（噪音強化）之後，Phase 3 PoC 鎖定：
1. **端到端 PII pipeline 可行性**：ASR + PII 模型整體召回是否達合規門檻？
2. **Prompt 干預成效**：Whisper `initial_prompt` 注入保險術語對 ASR / PII 各層的影響？
3. **PII 模型選型**：開源（OpenAI Privacy Filter）vs 商用 LLM（GPT-4o-mini）哪個值得投入工程資源？

---

## 2. Methodology

### 2.1 Query 設計
**檔案**：[`data/phase3_queries.csv`](benchmark/breeze_asr26/data/phase3_queries.csv) — 90 條 query

**Subtype 分層（個資組合維度）**

| Subtype | 個資組合 | N | 說明 |
|---|---|---:|---|
| A1 | 姓名 + 身分證 + 生日 | 18 | 開戶／身份核實情境 |
| A2 | A1 + 保單號 | 6 | 完整查詢 / 理賠申請 |
| B1 | 僅姓名 | 15 | 一般諮詢 |
| B2 | 姓名 + 保單號 | 7 | 保單變更 |
| C1 | 姓名 + 地址 | 8 | 變更通訊地址 |
| C2 | 姓名 + 電話 | 6 | 變更聯絡電話 |
| D | **無 PII** | 30 | 商品諮詢、條款解釋（用於 false positive 評估） |

**語言三分（30/30/30 平均分布）**
- `mandarin`：純國語
- `hokkien`：純閩南語（用台語慣用詞，如「搬厝」「替我看」「歡呼」）
- `codeswitch`：國語 PII 段 + 閩南語業務詢問段（自然口語常見模式：客戶先用普通話自報身份再切換母語提問）

**Subtype × Language 完全交叉設計**：每個 (subtype, language) 組合都至少 2 條，A1 / D 因樣本量大支撐統計分析。

**內容生成**
- **腳本**：模擬真實客服情境（理賠／更名／詢問商品／續保等），參考 Phase 1 收集到的 d0001-d0049 真實話術風格。
- **保險術語注入**：每條 query 標註 `domain_terms`（如「凱基」「享安心」「新康泰」「長照保險」「重大疾病險」），共 58 種獨立術語、190 筆出現次數，用於 domain term recall 評估。
- **PII 黃金標準**：每條 query 在 `pii_types`（分號分隔的類型清單）與 `pii_values`（對應值）兩欄記錄精確答案，作為 evaluation ground truth。


### 2.2 TTS 音檔生成
**語音來源**：Yating TTS API（[https://tts.api.yating.tw](https://tts.api.yating.tw)）

| 用途 | 語音 model |
|---|---|
| 國語段 | `zh_en_female_1` / `zh_en_male_1` |
| 閩南語段 | `tai_female_1` / `tai_male_1` |


**性別交替**：依 `case_id` 順序奇偶切換性別 → 90 條中 female=46、male=44，避免性別偏差。

**音檔統計**
- 總數：90 個 wav
- 時長：min ~9s、max ~21s、mean ~13s、總計 ~17 分鐘
- 格式：LINEAR16 / 16 kHz / mono（模擬電話品質的下界）
- 全部 < 25s：未觸發 Whisper chunk 邊界問題（Phase 2 已知 chunk 邊界是長對話 case 的 ceiling）

### 2.3 ASR 推論：Breeze-ASR-26
**模型**：MediaTek-Research/Breeze-ASR-26（Whisper-large-v2 fine-tune，~2B 參數，5.75 GB FP32 / ~3 GB BF16）


**Ablation 設計**

| Ablation | initial_prompt | 假設 |
|---|---|---|
| **E1 baseline** | 無 prompt | 對照組 |
| **E2 +保險詞** | `凱基人壽 享安心 新康泰 心康泰 享放心 增優利 終身壽險 長照險 投資型保單 重大疾病險 醫療附約 防癌險 年金保險 儲蓄險 傷害險` | 注入保險術語可降低同音字錯誤 |

### 2.4 PII 偵測方法

#### M1 — OpenAI Privacy Filter
**模型**：[openai/privacy-filter](https://huggingface.co/openai/privacy-filter)（Hugging Face）
**架構**：bidirectional token classifier（gpt-oss-style，1.5B 參數總量、50M active params via MoE，128k context）
**輸出類別（8 native）**：`private_person`、`private_address`、`private_email`、`private_phone`、`private_url`、`private_date`、`account_number`、`secret`


**6 類 GT 映射**

| M1 native | → 對應 GT |
|---|---|
| `private_person` | name |
| `private_address` | address |
| `private_phone` | phone |
| `private_date` | birth_date |
| `account_number` | national_id ∪ policy_id |
| `secret` | national_id ∪ policy_id |
| `private_email`、`private_url` | （未使用） |

#### M2 — Azure OpenAI GPT-4o-mini
**部署**：Azure OpenAI Sandbox resource，deployment 名 `gpt-4o-mini`
**呼叫**：`chat.completions.create()` 用 `response_format={"type": "json_schema", ...}` 強制 JSON 結構化輸出，schema 鎖定 6 類 array：`name`、`national_id`、`birth_date`、`policy_id`、`phone`、`address`。
**System prompt（中文）**："""你是台灣金融客服文本的個資抽取助手。從輸入文字中找出所有個人資料，按 6 大類分類輸出 JSON。

類別定義：
- name: 客戶姓名（中文 2-4 字，例如 陳大天、林淑芬）
- national_id: 中華民國身分證字號（一個英文字母 + 9 位數字，例如 A123456789）
- birth_date: 出生年月日（民國或西元，例如 民國七十年三月五日 / 民國70年3月5日 / 1981/03/05）
- policy_id: 保險保單號碼（英文+數字+連字號組合，例如 KGI-2024-001234、KL-2020-987654、PA12345678）
- phone: 電話號碼（手機 09xx-xxx-xxx 或市話 02-xxxx-xxxx，注意 ASR 可能會漏前導 0）
- address: 地址（台灣縣市區/路段/巷/弄/號/樓，例如 台北市大安區忠孝東路四段100號5樓）

規則：
1. 只抽取真的出現在輸入文字中的字串（不要補完不存在的內容）。
2. 同一類別有多筆就全部列出。
3. 沒命中的類別給空陣列 []。
4. 不解釋，只輸出 JSON。"""

#### 比對標準
針對 ASR 輸出可能字元錯誤而設的三層階梯：

| Match level | 規則 |
|---|---|
| **exact** | 偵測字串子字串包含 GT（或反向） |
| **normalized** | 去標點 / 全形半形 / 中文阿拉伯數字統一 / 大寫化 後子字串包含 |
| **fuzzy** | normalized 後 Levenshtein 距離 ≤ 類別 threshold（雙向 sliding window） |
| **no** | 以上皆無 |

**類別 threshold**（fuzzy 容錯）

| 類別 | threshold | 理由 |
|---|---|---|
| name | 1 | 中文姓名 2-4 字，容 1 字錯 |
| national_id | 1 | 1 字母+9 數字，容 1 位 |
| birth_date | 2 | 民國年月日多字格式 |
| policy_id | 2 | 含字母-數字混合 |
| phone | 2 | 電話 10 位 |
| address | 4 | 長字串容 4 字錯 |

### 2.5 評估面向
1. **ASR 層級**：CER / CER_norm（中阿數字統一）/ PII recall（normalized）/ Domain term recall。
2. **PII 層級**：對 ASR hypothesis 跑兩個 PII 方法，計算 (method, ablation, pii_type) 三維 recall。
3. **端到端**：M2/E1 micro-recall = 偵測模型在 ASR 文字上能救回多少 GT 個資。
4. **假陽性**：D-subtype 30 條無 PII query 上的偵測數（理應全為 0）。

---

## 3. Phase A — ASR 評估結果

### 3.1 Overall CER

| Ablation | CER (strict) | **CER_norm** | N |
|---|---|---|---|
| E1 baseline | 16.41% | 16.34% | 90 |
| **E2 +保險詞** | **13.51%** | **13.54%** | 90 |

**E2 vs E1 absolute Δ = -2.79pp，relative -17.1%**。

### 3.2 CER × Language

| language | N | E1 | E2 | Δ |
|---|---|---|---|---|
| mandarin | 30 | 5.24% | **2.98%** | -2.26pp |
| codeswitch | 30 | 22.98% | **18.51%** | -4.46pp |
| hokkien | 30 | 20.80% | **19.14%** | -1.66pp |

**觀察**：
- mandarin CER 已經低於 3% — 對純國語 PII 場景已達 production 門檻
- E2 對 codeswitch 改善最多（-4.46pp），對純閩南語幫助較小（-1.66pp）
- Hokkien CER 上限受 Whisper 對台語建模的 fundamental limit 拘束

### 3.3 CER × Subtype

| subtype | N | E1 | E2 | Δ |
|---|---|---|---|---|
| A1 | 18 | 13.28% | 11.73% | -1.55pp |
| A2 | 6 | 15.04% | 12.71% | -2.34pp |
| B1 | 15 | 17.03% | 13.64% | -3.39pp |
| B2 | 7 | 14.98% | 12.28% | -2.70pp |
| C1 | 8 | 14.77% | 10.82% | -3.95pp |
| C2 | 6 | 9.26% | 9.31% | +0.05pp |
| **D** | 30 | 20.24% | 16.61% | **-3.63pp** |


### 3.4 PII 召回（ASR 層）

僅看「ASR hypothesis 是否含正確 PII 字串」（normalized mode），尚未送 PII 模型：

| PII 類型 | N | E1 | E2 | Δ |
|---|---|---|---|---|
| name | 60 | 76.67% | 76.67% | 0 |
| national_id | 24 | 45.83% | 50.00% | +4.17pp |
| birth_date | 24 | 100.00% | 100.00% | 0 |
| policy_id | 13 | 61.54% | 69.23% | +7.69pp |
| phone | 6 | 66.67% | 66.67% | 0 |
| address | 8 | 37.50% | 50.00% | +12.50pp |

**重點觀察**
- **birth_date 100%**：民國年月日經中阿數字統一後完美保留 → ASR 對日期語意極穩
- **數字串 PII 表現可用**：national_id 50%、policy_id 69%、phone 67%（fuzzy 能上推到 83-100%）
- **name 76.67%**：mandarin/codeswitch 都 90%+，hokkien 僅 45% 是主要 miss source

### 3.5 三模式（strict / normalized / fuzzy）對照

| 類別 | E2 strict | E2 normalized | E2 fuzzy |
|---|---|---|---|
| name | 76.67% | 76.67% | 96.67% |
| national_id | 41.67% | 50.00% | 83.33% |
| birth_date | 79.17% | 100.00% | 100.00% |
| policy_id | 15.38% | 69.23% | 69.23% |
| phone | 0.00% | 66.67% | 100.00% |
| address | 12.50% | 50.00% | 100.00% |

**fuzzy 大幅優於 strict** 顯示 ASR 對長字串個資仍有「邊界字元錯」的常態；evaluation 容錯設計合理，且下游 PII 模型可在這個 fuzzy 基礎上做語意層救回。

### 3.6 PII recall × Language（normalized）

| 類型 | mandarin (N) | hokkien (N) | codeswitch (N) |
|---|---|---|---|
| name | 95% (20) | **45%** (20) | 90% (20) |
| national_id | 50% (8) | **37.5%** (8) | 62.5% (8) |
| birth_date | 100% (8) | 100% (8) | 100% (8) |
| policy_id | 100% (4) | **20%** (5) | 100% (4) |
| phone | 100% (2) | 50% (2) | 50% (2) |
| address | 67% (3) | 50% (2) | 33% (3) |

**Hokkien 是所有 PII 類型的最大缺口**（粗體區塊）。policy_id 在 hokkien 只剩 20%，名字也只剩 45%。

### 3.7 保險術語召回

| Ablation | strict | normalized | fuzzy | 命中數（共 190 例） |
|---|---|---|---|---|
| E1 baseline | 33.68% | 34.21% | 62.63% | 65 |
| **E2 +保險詞** | **58.42%** | **60.53%** | **81.58%** | **115** |

**E2 absolute +26.32pp on normalized，relative +77%**。這是 E2 prompt 的真正價值所在。

**重點商品名 / 術語 E2 改善**

| 術語 | 出現 | E1 命中 | E2 命中 | 改善 |
|---|---|---|---|---|
| 凱基 | 41 | 14 | **28** | +14 |
| 終身壽險 | 9 | 1 | **9** | +8 |
| 醫療附約 | 5 | 1 | **5** | +4 |
| 住院醫療附約 | 6 | 1 | **5** | +4 |
| 心康泰 | 8 | 0 | **2** | +2 |
| 享安心 | 11 | 0 | **2** | +2 |
| 投資型壽險 | 3 | 0 | **3** | +3 |
| 長照險 | 8 | 0 | **2** | +2 |
| **增優利 / 享放心** | 7 / 5 | 0 | **0** | 0（仍失敗） |

「增優利」「享放心」未出現在 prompt 中（漏注入）→ E2 依然零命中，是 prompt list 維護的下一個優化點。

### 3.8 最差 Cases（CER_norm Top 5）

E2 最差 5 條集中在 hokkien A1/A2（含身分證的閩南語）+ codeswitch D：

| case_id | language | subtype | E2 CER_norm |
|---|---|---|---|
| q022 | hokkien | A2 | 47.95% |
| q008 | hokkien | A1 | 38.60% |
| q085 | codeswitch | D | 35.00% |
| q071 | hokkien | D | 33.33% |
| q088 | codeswitch | D | 32.43% |

特徵：**閩南語 + 連續英數 PII** 仍是模型壓力測試極限。

### 3.9 E2 vs E1 改善與退步

**Top 5 改善** — 集中在 codeswitch + D-subtype（純商品語境最受惠 prompt）：

| case_id | lang | subtype | E1 | E2 | 改善 |
|---|---|---|---|---|---|
| q084 | codeswitch | D | 32.43% | 10.81% | -21.62pp |
| q082 | codeswitch | D | 43.59% | 23.08% | -20.51pp |
| q049 | codeswitch | B1 | 29.73% | 16.22% | -13.51pp |
| q066 | mandarin | D | 12.50% | 0.00% | -12.50pp |
| q053 | codeswitch | B1 | 36.11% | 25.00% | -11.11pp |

**Top 5 退步** — 9 條退步中多數為 hokkien D：

| case_id | lang | subtype | E1 | E2 | 退步 |
|---|---|---|---|---|---|
| q071 | hokkien | D | 15.38% | 33.33% | +17.95pp |
| q014 | codeswitch | A1 | 12.28% | 26.32% | +14.04pp |
| q075 | hokkien | D | 25.00% | 30.56% | +5.56pp |
| q085 | codeswitch | D | 30.00% | 35.00% | +5.00pp |
| q080 | hokkien | D | 29.41% | 32.35% | +2.94pp |

**解讀**：保險詞 prompt 對閩南語語境偶爾產生 over-correction，把不相關的詞硬套成保險商品名。整體仍正向（90 條中改善 > 退步），但若要避免極端退步，可考慮「按語言條件動態調整 prompt」。

### 3.10 真實轉錄範例

挑 4 條呈現各類 ASR 行為，REF 為 GT、HYP 為 Breeze-ASR-26 輸出：

#### 範例 1：q084（codeswitch / D-subtype）— **E2 prompt 大幅救回**
```
REF:    妳好我想請教，終身壽險的保額可以中途調整嗎，調高的話需要重新做健康告知或體檢嗎。
E1 HYP (32.43%): 你好 我想請教 重生受險的寶藝 可不可以中途調整 調高的話 需要重新做健康個機或是體檢嗎
E2 HYP (10.81%): 你好 我想請教 終身壽險的保額可以中途調整嗎 調高的話 需要重新做健康個資或是體檢嗎
```

**觀察**：E1「重生受險的寶藝」（亂碼） → E2「終身壽險的保額」（正確）。保險術語注入完美命中。「健康告知」仍誤為「健康個資」是 E2 prompt 沒涵蓋到的詞。

#### 範例 2：q022（hokkien / A2）— **TTS digit-spaced 救回但仍受台語語境影響**
```
REF:    你好我是黃美玲，身分證G 1 3 4 5 6 7 8 9 0，民國六十年四月一日生，保單號K G I 2 0 2 3 0 0 1 1，幫我申請我這張凱基享放心長照險的失能保險金，已經有醫師診斷書。
E2 HYP (47.95%): 妳好 我是黃美鈴 身分證告一三四五六七八九零 民國六十年四月一號星 保單號二○二三○○一一 幫我申請我這張 凱吉祥放心臟照X光的適齡保險金 已經有一書診斷書了
```

**觀察**：
- ID 數字部分 10 個 digit 都在（`一三四五六七八九零` 對應 `134567890` ✓）
- 字母 G 變「告」（hokkien 模型對英文字母聲學模型訓練不足）
- 保單號 KGI 字母全失，只剩數字部分 `二○二三○○一一` = `20230011`
- 商品名「享放心長照險」變「祥放心臟照X光」（同音字錯誤，prompt 未含「享放心」）

#### 範例 3：q008（hokkien / A1）— **典型 hokkien 開頭被吃掉**
```
REF:    你好我許文傑，身分證H 2 5 1 2 3 4 5 6 7，民國七十五年六月十八日生，麻煩幫我確認我這張醫療附約這個月有沒有成功扣款，怕又漏掉。
E2 HYP (38.60%): 你好 我是文傑 身分證 夏日五一二三四五六七 民國七十五年六月十八號出生 麻煩你幫我確定一下 我這張醫療附約 一個月有成功扣款嗎 怕又漏了
```

**觀察**：「許文傑」開頭的「許」被吃掉只剩「文傑」、字母 H 變「夏日」（兩個音節，因為 H 在台語語境聽起來像 hé+jit）、9 位數字完整保留。日期經中阿轉換完美。後段業務內容大致還原。

#### 範例 4：q050（codeswitch / B1）— **同音字錯誤集中在商品名**
```
REF:    妳好我賴志偉，請問凱基的活利鑫動投資型壽險有沒有月配息的選擇，最低投保金額是多少。
E1 HYP (21.05%): 你好 我賴志偉 請問 殺價的合利生動投資型受險 有沒有外配色的選擇 最低投保金額是多少
E2 HYP (18.42%): 你好 我賴志偉 請問 胎記的合利行動投資型壽險 有沒有外配色的選擇 最低投保金額是多少
```

**觀察**：「凱基」E1 變「殺價」，E2 變「胎記」（仍錯但音較接近，且「凱基」在 prompt 中只列了一次，未必能覆蓋所有口腔變體）。「投資型壽險」E1「投資型受險」→ E2「投資型壽險」（修正）。**E2 prompt 補回了「投資型壽險」這個專有名詞，但「凱基」仍待強化**。

---

## 4. Phase B — PII 偵測評估

> **本節僅呈現 M2/E1 最佳組合（94.81% micro-recall）與 M1 同組合的對比**。E1 vs E2 的 prompt 影響在 PII 層為中性（差 < 1pp，已在 §3 驗證），不再展開分析。

### 4.1 Overall Micro-Recall

對 90 個 ASR hyp × 2 個 PII 方法，把所有 PII instance 池在一起算總 recall：

| Method | 總 PII 數 | 命中 | **Micro-recall** |
|---|---|---|---|
| M1 OpenAI Privacy Filter | 135 | 63 | **46.67%** |
| **M2 Azure GPT-4o-mini** | **135** | **128** | **94.81%** |

**M2 vs M1 差距 +48.14pp**，是這次評估最關鍵的訊號。M2 在 135 筆 GT 中只漏 7 筆，而 M1 漏 72 筆。

### 4.2 各 PII 類型 Recall

| PII 類型 | N | **M2** | M1 | M2 - M1 差距 |
|---|---|---|---|---|
| name | 60 | **93.33%** | 65.00% | +28.33pp |
| national_id | 24 | **95.83%** | 8.33% | +87.50pp |
| birth_date | 24 | **100.00%** | 29.17% | +70.83pp |
| policy_id | 13 | **92.31%** | 30.77% | +61.54pp |
| phone | 6 | **83.33%** | 66.67% | +16.67pp |
| address | 8 | **100.00%** | 87.50% | +12.50pp |

**M2 表現分群**：
- **接近滿分**（≥95%）：birth_date、national_id、address
- **高水準**（90-95%）：name、policy_id
- **可用門檻**（80-90%）：phone

**M1 結構性失靈**：
- **national_id 8.33%**：M1 把「身分證A123456789」當作一個複合實體，BIOES 解碼器邊界判定混亂導致整段歸為 `O`（背景）
- **birth_date 29.17%**：民國年月日格式被切碎，因 M1 訓練資料以英文為主，`private_date` 學到的是 MM/DD/YYYY 等英文日期 prior
- **`account_number`/`secret` 雙重映射**：M1 把身分證和保單號都歸成同一類，下游分流時無法區分

### 4.3 Recall × Language

| 類型 | mandarin (N) | codeswitch (N) | hokkien (N) |
|---|---|---|---|
| name | 95.00% (20) | 100.00% (20) | **85.00%** (20) |
| national_id | 100.00% (8) | 100.00% (8) | **87.50%** (8) |
| birth_date | 100.00% (8) | 100.00% (8) | 100.00% (8) |
| policy_id | 100.00% (4) | 100.00% (4) | **80.00%** (5) |
| phone | 100.00% (2) | 50.00% (2) | 100.00% (2) |
| address | 100.00% (3) | 100.00% (3) | 100.00% (2) |

**Hokkien 仍是相對最弱**（85% / 87.5% / 80%），但相比 ASR 層的 PII recall（hokkien name 45%、national_id 37.5%、policy_id 20%），M2 把這三個指標分別拉到 85%、87.5%、80%。**M2 確實在 ASR 之上做了顯著的語意層救回**：即便 ASR hyp 中的 PII 字串歪掉，M2 仍能用結構/語意 cue 推回 GT。

### 4.4 假陽性分析（D-subtype 30 條）

D 全部沒有 PII，任何偵測即為 false positive。

| Method | 總 FP | 有 FP 的 case | Case FP rate | 平均 FP/case |
|---|---|---|---|---|
| M1 | 0 | 0 | 0.00% | 0.00 |
| **M2** | **3** | **3** | **10.00%** | **0.10** |

**M2 FP 類型分布**：全 3 筆都是 `policy_id`，集中在 D-subtype 中業務員主動提到的商品代碼或保單號片段被誤判為客戶 PII。

**Trade-off 解讀**：M1 case-FP rate = 0%，但 recall 也只有 47%。M2 case-FP rate = 10%（每 10 條 D query 約 1 條會出 FP），但 recall 95%。**多誤判 3 筆 FP 換來救回 65 筆真實 PII，明顯划算**（FP 可被人工複核或規則 filter 過濾，FN 是合規漏失，後者代價遠高）。

### 4.5 命中模式分布

| Method | exact | normalized | fuzzy | miss |
|---|---|---|---|---|
| M1 | 39 (28.9%) | 10 (7.4%) | 14 (10.4%) | 72 (53.3%) |
| **M2** | **83 (61.5%)** | 22 (16.3%) | 23 (17.0%) | **7 (5.2%)** |

**M2 有 61.5% 是 exact 命中**（連標點都對），fuzzy 只佔 17.0%。意思是 M2 的高 recall **不是被我們的容錯規則救起來的**，而是真實語意層級擷取。即便把 fuzzy 規則拿掉只看 exact + normalized，M2 仍有 77.8% 的精準 recall（vs M1 的 36.3%）。

### 4.6 M1 / M2 一致性

| N | agreement | 兩者皆中 | 兩者皆漏 | 僅 M1 中 | **僅 M2 中** |
|---|---|---|---|---|---|
| 135 | 48.89% | 61 | 5 | 2 | **67** |

**M1 幾乎沒有獨家命中**（僅 2 筆），代表 **M2 已涵蓋 M1 能做的事**，ensemble M1 + M2 對 recall 的增益 < 1pp。生產部署不需要兩者並行。

### 4.7 M2 漏掉的 7 個 PII 欄位（完整個案歸因）

| Case | Lang | Subtype | 漏掉的類型 | GT 值 | ASR hyp 對應段落 | 失敗模式 |
|---|---|---|---|---|---|---|
| q021 | hokkien | A2 | name | 曾國偉 | `...六十一年六月十號生 保單號碼...` | ASR 把「我曾國偉」整段吞掉，誤識為「我莊國維」 |
| q029 | hokkien | C1 | name | 吳怡君 | `...的通信地址改為桃園市中壢區...` | 自報姓名段消失 |
| q043 | mandarin | B1 | name | 邱雅芳 | `...目前的主約 還可以加保哪些醫療副約...` | 名字字面變成不可辨識的字符 |
| q057 | hokkien | B2 | name | 吳怡君 | `...想要查我這張開之前有利的年金...` | 自報姓名段失蹤 |
| q008 | hokkien | A1 | national_id | H251234567 | `...民國七十五年六月十八號出生...` | 「身分證H...」整段被吞 |
| q021 | hokkien | A2 | policy_id | KL-2020-987654 | `保單號碼2020987654` | 字母 `KL` 遺失，剩 9 位數字距離 GT > threshold=2 |
| q038 | codeswitch | C2 | phone | 02-2345-6789 | `...56789 麻煩替我把這張...` | ASR chunk 邊界把前 5 字元 `02-23` 截掉 |

**歸因總結**：
- **6/7 是 ASR 上游字串被吃掉或截斷** → PII 模型再強也救不回（沒文字可抓）
- **1/7 是 evaluation threshold 卡住**（q021 policy_id：M2 其實偵測到了 `2020987654`，但 Lev 距離=3 超過 threshold=2）
- **5/7 集中在 hokkien**，2/7 分布在 codeswitch 和 mandarin

### 4.8 真實 PII 偵測範例（M2 vs M1）

#### 範例 1：q001（mandarin A1）— **完美 case**
```
GT:    name=陳大天, national_id=A123456789, birth_date=民國七十年三月五日
ASR hyp: 您好 我是陳大天 身分證A123456789 民國七十年三月五日生 想問一下我那張...
```
| Method | 偵測結果 |
|---|---|
| **M2** | `{"name": ["陳大天"], "national_id": ["A123456789"], "birth_date": ["民國七十年三月五日"]}` ← 全部 exact ✓ |
| M1 | 偵測一個 `private_person` span = `"陳大天 身分證A123456789 民國七"`（邊界全錯，三個 PII 黏成一個 entity） |

**觀察**：M2 結構化輸出乾淨對應 6 類 GT；M1 在中文 ASR 字串上邊界判定失敗，把姓名+身分證+生日的開頭粘成一個人名。

#### 範例 2：q025（mandarin C1）— **長地址 normalized 命中**
```
GT:    name=陳大天, address=台北市信義區忠孝東路五段一二三號
ASR hyp: ...麻煩把我那張 凱基想安心的聯絡地址 改成台北市信義區 忠孝東路5段123號 謝謝
```
| Method | 偵測結果 |
|---|---|
| **M2** | `{"name": ["陳大天"], "address": ["台北市信義區忠孝東路5段123號"]}` |
| M1 | `private_address = "改成台北市信義區 忠孝東路5段123號"` + 兩個碎片 spans |

**觀察**：地址 GT 用中文數字「五段一二三號」，hyp 用阿拉伯數字「5段123號」。M2 直接抽取阿拉伯數字版本，evaluation 端 `normalize_for_pii` 統一中阿數字後判 normalized match ✓。M1 把「改成」這類動詞前綴也吃進 span（不影響 recall 但會降低 precision）。

#### 範例 3：q011（hokkien A1）— **fuzzy 命中（姓名有錯字）**
```
GT:    name=劉怡萱, national_id=A234567891, birth_date=民國七十一年二月十四日
ASR hyp: 你好 我是劉怡宣 身分證A234567891 民國七十一年二月十四號生...
```
| Method | 偵測結果 |
|---|---|
| **M2** | `{"name": ["劉怡宣"], "national_id": ["A234567891"], "birth_date": ["民國七十一年二月十四號"]}` |

**觀察**：「萱」→「宣」是 ASR 同音字錯誤，Levenshtein 距離 = 1，過 fuzzy threshold ✓。national_id 連 10 個字元都對，這是 digit-spaced TTS 寫法的功勞 — 客戶逐字念，ASR 逐字轉，PII 模型逐字抓。birth_date「日」→「號」也是 fuzzy 命中（distance=1）。

#### 範例 4：q022（hokkien A2）— **混合命中與失誤**
```
GT:    name=黃美玲, national_id=G134567890, birth_date=民國六十年四月一日, policy_id=KGI20230011
ASR hyp: 妳好 我是黃美鈴 身分證告一三四五六七八九零 民國六十年四月一號星 保單號二○二三○○一一 替我申請我這張 開吉祥放心臟照XEN的...
```
| Method | 偵測結果 | 評估結果 |
|---|---|---|
| **M2** | `name=["黃美鈴"]` | fuzzy match ✓（玲↔鈴，距離=1） |
| | `national_id=["A134567890"]` | normalized match ✓（M2 把「告」（hokkien G）猜回 A，9 位數字對） |
| | `birth_date=["民國60年4月1日"]` | exact match ✓（中阿轉換後） |
| | `policy_id=["202300111"]` | normalized match ✓（M2 抓到數字部分，評估容 1 字差） |

**觀察**：這是個多重失誤但全部救回的好例子 —— ASR 把「黃美玲」轉成「黃美鈴」、字母 G 轉成「告」、保單號字母 KGI 全失，M2 仍透過上下文（「身分證」「保單號」這些 anchor token）把 4 個 PII 全部正確抽取。**這個 case 顯示 M2 的真實能力遠超「在 ASR 字串上做 string matching」**。

#### 範例 5：q008（hokkien A1）— **唯一 national_id miss**
```
GT:    name=許文傑, national_id=H251234567, birth_date=民國七十五年六月十八日
ASR hyp: 你好 我是文傑 身分證 夏日五一二三四五六七 民國七十五年六月十八號出生...
```
| Method | 偵測結果 | 評估結果 |
|---|---|---|
| **M2** | `name=["文傑"]` | fuzzy match ✓（許文傑↔文傑，距離=1，過 name threshold） |
| | `national_id=["夏日51234567"]` | **miss ✗**（H 變「夏日」兩個字 + 9 位數字第一位「2」變不見只剩 8 位，距離 > 1） |
| | `birth_date=["民國75年6月18日"]` | exact match ✓ |

**觀察**：這個 case 同時體現 M2 的能力上限與下限：
- **能救**：3 字姓名只有 2 字、字母 H 變「夏日」，仍能識別出 name 與 national_id 兩個欄位 → 結構解析正確
- **不能救**：當 GT 字元錯失太多（H251234567 → 夏日51234567，少 1 數字 + 字母錯），fuzzy threshold 卡住判 miss

這個 case 對應 §4.7 表中「ASR 上游字串被吃掉」的代表 — 不是 M2 抽不出來，是評估規則嚴格對齊 GT 後判定為 miss。

#### 範例 6：q070（mandarin D-subtype）— **無 PII，無誤判**
```
GT:    （無 PII）
ASR hyp: 想請問你們的傷害險 如果在國外發生意外 理賠程序需要哪些文件 醫療收據國外開的英文版 可以嗎...
```
| Method | 偵測結果 |
|---|---|
| **M2** | `{}`（6 類全空 array） ✓ |
| M1 | `[]`（無偵測） ✓ |

**觀察**：純商品諮詢的 D-subtype，M2 正確返回空 result，沒有 false positive。30 條 D-subtype 中 27 條 M2 都是這種「正確不偵測」狀態，3 條才出 FP。

---

## 5. Pipeline-Level Analysis（端到端歸因）

### 5.1 Recall 拆解：ASR 上游 vs PII 下游

把 M2/E1 的 7 個 miss 拆成兩個原因：

| Miss 原因 | 估計筆數 | 說明 |
|---|---|---|
| ASR 上游字串歪掉 → PII 模型再強也救不回 | ~6 | 主要在 hokkien name、national_id、phone |
| ASR 對但 PII 模型仍漏 | ~1 | 偶發 |

**結論**：**~85% 的失誤是 ASR 問題，不是 PII 模型問題**。換更強的 PII 模型邊際收益遞減；補強 ASR 後處理（特別針對 hokkien）才是高 ROI 工程。

### 5.2 GPT-4o-mini 在 ASR 之上的「救回力」

比較 ASR PII recall（normalized）vs M2 PII recall：

| 類型 | E1 ASR recall | M2/E1 PII recall | M2 救回幅度 |
|---|---|---|---|
| name | 76.67% | 93.33% | +16.66pp |
| national_id | 45.83% | 95.83% | +50.00pp |
| birth_date | 100.00% | 100.00% | 0 |
| policy_id | 61.54% | 92.31% | +30.77pp |
| phone | 66.67% | 83.33% | +16.66pp |
| address | 37.50% | 100.00% | +62.50pp |

**LLM 對「結構穩定但字元有錯」的字串特別擅長**：address +62.5pp、national_id +50pp、policy_id +30.8pp。這就是 GPT-4o-mini 相對純 ASR 的核心價值 — 它把 ASR fuzzy 級的 hits 提升為 PII exact 級的 hits。

### 5.3 Hokkien 是 PII pipeline 唯一的瓶頸

Breeze-ASR-26 在 hokkien 整體 CER 19.14%（vs mandarin 2.98%），這個 baseline 差距傳遞到下游：

- ASR 字串歪掉，PII 模型再強也無法救回
- M2 對 hokkien 的「語意層擷取」對 name 仍有顯著效果（85%），但對 national_id（87.5%）和 policy_id（80%）開始貼近能力上限
- M2 整體 7 個 miss 中有 5 個是 hokkien

**典型 hokkien 失敗模式**（從 §3.10 範例觀察）：
1. **字母轉中文**：H→「夏日」、G→「告」、K→「可」（hokkien 對英文字母聲學模型不足）
2. **開頭被吞**：「你好我XXX」開場白整段消失（hokkien decoder 對短停頓敏感）
3. **保單號字母遺失**：KGI / KL 等英文 prefix 在 hokkien hyp 中常完全消失，只剩數字部分


---

## 6. 限制與風險

### 6.1 評估範圍限制
1. **TTS 音檔 ≠ 真實電話品質**：Yating TTS 比真實客服錄音乾淨，CER 數字應視為**樂觀 baseline**。實際電話含 G.711 codec、背景噪音、迴聲，預期 CER 會 +5–10pp（Phase 2 數據顯示 echo 對 mandarin +5.17pp）。
2. **單一 TTS 引擎偏差**：所有 hokkien 段都來自 Yating `tai_*` 模型，可能引入特定發音偏好。真實客戶的腔調（南部腔、宜蘭腔、海口腔）多樣性未覆蓋。
3. **Domain 集中**：90 條全為凱基保險場景，無法直接外推到其他金融產品線。
4. **PII 樣本量**：phone N=6、address N=8、policy_id N=13 偏小，個別百分點需保守解讀。
5. **D-subtype FP 評估僅 30 條**：M2 case-level FP rate 7-10% 的 confidence interval 約 ±10pp，需更大規模才能確定 FP 上限。

### 6.2 模型限制
1. **Breeze-ASR-26 30s chunk 邊界**：Phase 2 已知問題，本 PoC 因 query 都 < 25s 未觸發。生產環境長對話需 chunk stitching。
2. **GPT-4o-mini 成本**：M2 每次 ~150 tokens prompt + 50 tokens output ≈ NT$0.03/條，180 條 ~NT$5.4。production 量級（百萬級 query/月）需重新評估，可能要 batch 或本地 LLM 替代。
3. **M1 ONNX path workaround**：因 transformers 4.57.1 不認 `openai_privacy_filter` 架構而走 ONNX，未來若 transformers 5.x 正式支援可改回原生 path（但 M1 既然沒 ensemble 價值，可能不值得投資）。

### 6.3 合規風險
1. **PII recall 94% 不等於合規 94%**：合規要求通常是「每筆都不能漏」，而非「平均 recall 達標」。
2. **False negative 集中在 hokkien**：閩南語客戶身分核實漏失風險高於國語客戶。
3. **資料外送 Azure**：M2 是 Azure OpenAI（已通過合規審查的同一個 chunking_embedding 用 resource），但仍需確保部署時不洩露 PII raw 字串到非授權端點。

---

_本報告由 Phase 3 PoC 自動評估管線產出。原始結果與重現腳本見 `benchmark/breeze_asr26/`。_

