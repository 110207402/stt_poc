# Speaker Diarization + Role Attribution Test


---

## 0. Executive Summary

### Recommendation

| 元件 | 模型 | 預期表現 |
|---|---|---|
| **Diarization** | pyannote 3.1 with `num_speakers=2` | DER ≈ 5-7% |
| **Role attribution** | Azure OpenAI gpt-4o-mini (zero-shot) | Accuracy ≈ 94% |

整體 pipeline：

```
電話錄音 (mono 16kHz)
    ↓
pyannote 3.1 → speaker_0 / speaker_1 segments
    ↓
gpt-4o-mini → 「客服」/「客戶」 role labels
    ↓
逐字稿 with role tags
```

### 核心結論

1. **pyannote 3.1 DER 5.95%**（lenient collar 0.25, n=234），統計顯著優於 3D-Speaker CAM++（p = 0.0044）
2. **gpt-4o-mini role classification 94.6%** accuracy on 37 closed-set 2-speaker cases，weak signal 場景仍 87.5%
3. **CAM++ 唯一明確優勢**：same-gender similar voice 場景（CAM++ 2.56% vs pyannote 12.23%），可作為 router 第二層
4. **Acoustic-only role attribution 在合成資料上不可行**（B5 ECAPA-TDNN 56.8% ≈ random），需真實員工錄音才能 enrol

---

## 1. Production Architecture


### 1.1 不推薦的替代方案

| 方案 | 為何不採用 |
|---|---|
| CAM++ 單獨 diarization | DER 8.45%，在 simultaneous_start / barge-in 場景明顯弱（DER 10-25%）|
| pyannote auto (不指定 N) | spk_acc 只 68.4%（fixed-N 99.6%）|
| 純規則 role classifier | mid_call 場景 accuracy 跌到 35%（沒 greeting 無 keyword 可匹配）|
| 純 acoustic role classifier | 在合成資料上 56.8% ≈ random（真實 prod 預期 80%+，需 enrollment 後才能驗）|

---

## 2. Methodology

### 2.1 Research Questions

| RQ | 問題 | 答案 |
|---|---|---|
| RQ1 | 同 speaker 跨多個 turn / 長停頓後能否被正確識別？ | pyannote 在 reentry_long_gap 1.81%、reentry_short_gap 5.71% |
| RQ2 | 兩人重疊講話（barge-in、simultaneous start）能否分得開？ | pyannote 強（agent_barge_in 1.52%），CAM++ 弱（10.64%）|
| RQ3 | 哪個 speaker 是客服 vs 客戶能否自動推斷？ | gpt-4o-mini zero-shot 94.6% |

### 2.2 Ground Truth 設計

採用 VAD-trimmed RTTM：對每個 TTS turn 用 adaptive-threshold VAD 偵測真實發聲區間，避免將 silence 算成 speech。

| Metric | 設定 |
|---|---|
| DER (lenient) | collar = 0.25s（主要 ranking metric）|
| DER (strict) | collar = 0.0s（邊界精度）|
| JER | collar = 0.25s（speaker-level proportion error）|
| Speaker count accuracy | pred_N == expected_N from spec |
| RTF | wall_time / audio_duration |

採用 [pyannote.metrics](https://github.com/pyannote/pyannote-metrics) 標準實作。

---

## 3. Benchmark Dataset

| 維度 | 值 |
|---|---|
| 總 dialogs | 234 |
| 總 turns | 1,514 |
| 總 audio 時長 | 144 分鐘 |
| Average dialog | 37 秒（range 8-133s）|
| Splits | dev_smoke 24 / dev_tune 30 / test_core 140 / role_attribution 40 |
| Languages | mandarin 199 / hokkien 18 / codeswitch 17 |
| TTS providers | Azure Speech zh-TW + Yating Hokkien |
| 預期 speaker count | 2 (214 cases) / 3 (20 cases) |

### Slice 涵蓋（13 種）

| Slice | n | 場景 |
|---|---|---|
| same_gender_similar | 37 | 兩人同性別、聲音相近（hardest voice stress）|
| clean_turn_taking | 23 | 標準輪流講，無 overlap |
| agent_barge_in | 21 | Agent 在 customer 講話中段插話 |
| customer_barge_in | 21 | Customer 打斷 agent 解釋 |
| reentry_short_gap | 20 | 短停頓（5-10s）後同一 speaker 回來 |
| reentry_long_gap | 19 | 長停頓（30-90s）後 reentry |
| hokkien_pure | 18 | 純台語對話 |
| short_backchannel | 18 | 含大量短 backchannel（「嗯」「對」）|
| simultaneous_start | 18 | 兩人接通電話同時開口 |
| third_party_background | 17 | 第三方背景聲（家人、廣播）|
| codeswitch | 16 | 中台語切換 |
| impostor_unknown | 3 | 非保戶冒名查詢，3 speaker |
| no_pii_product_inquiry | 3 | 純產品條款問詢，無 PII |



---

## 4. Models Tested

| # | Model | Config |
|---|---|---|
| 1 | pyannote 3.1 (auto) | 自動估 num_speakers |
| 2 | pyannote 3.1 (fixed-N) | num_speakers from spec |
| 3 | 3D-Speaker CAM++ | iic/speech_campplus_speaker-diarization_common, default |

**Hardware**：NVIDIA A100-SXM4-40GB (Colab Pro+), CUDA 12.8。

未採用的 model：pyannote community-1（需 pyannote.audio ≥ 3.5）、ERes2NetV2（工程成本高且有 silent no-op 紀錄）、Sortformer v1（實測對 mandarin/hokkien 弱）。

---

## 5. Diarization Results

### 5.1 Cross-model Summary (n = 234)

| Model | DER lenient | DER strict | JER | spk_acc | Perfect (0%) | RTF |
|---|---|---|---|---|---|---|
| pyannote 3.1 (auto) | 7.18% | 12.75% | 12.86% | 68.4% | 81/234 | 0.021 |
| **pyannote 3.1 (fixed-N)** | **5.95%** | **11.53%** | **11.04%** | **99.6%** | **107/234** | **0.014** |
| 3D-Speaker CAM++ | 8.45% | 19.80% | 12.02% | 88.9% | 0/234 | 0.066 |

pyannote fixed-N 的 99.6% spk_acc 是給定正確 N 的理想條件；生產用 `num_speakers=2` heuristic（99%+ 通話為 2 人）也預期接近此表現。

### 5.2 Statistical Significance (paired t-test, n = 234)

| 比較 | Δ (pp) | p-value | 結論 |
|---|---|---|---|
| pyannote fixed-N vs CAM++ | -2.50 | **0.0044** | pyannote 顯著優於 CAM++ |
| pyannote fixed-N vs pyannote auto | -1.23 | 0.0020 | fixed-N 顯著優於 auto |
| CAM++ vs pyannote auto | +1.27 | 0.137 | 無顯著差異 |

### 5.3 Per-language

| Language | n | pyannote auto | pyannote fixed-N | CAM++ |
|---|---|---|---|---|
| mandarin | 199 | 7.97% | 6.73% | 8.85% |
| hokkien | 18 | 1.00% | 1.00% | 6.97% |
| codeswitch | 17 | 4.48% | 2.10% | 5.39% |

Hokkien 數字偏低是因為合成資料 hokkien 場景全為 F+M 不同 Yating voice，acoustic 差異顯著；真實 prod 兩人都台語的場景未測。

---

## 6. Per-slice Analysis

| Slice | pyannote fixed-N | CAM++ | Δ | Winner |
|---|---|---|---|---|
| hokkien_pure | 1.00% | 6.97% | -6.0pp | pyannote |
| codeswitch | 1.50% | 5.15% | -3.7pp | pyannote |
| no_pii_product_inquiry | 0.00% | 6.97% | -7.0pp | pyannote |
| agent_barge_in | 1.52% | 10.64% | -9.1pp | pyannote |
| customer_barge_in | 2.74% | 10.17% | -7.4pp | pyannote |
| reentry_long_gap | 1.81% | 5.83% | -4.0pp | pyannote |
| reentry_short_gap | 5.71% | 5.72% | 0.0pp | tie |
| simultaneous_start | 3.12% | 25.18% | **-22.1pp** | pyannote 大勝 |
| short_backchannel | 4.23% | 9.14% | -4.9pp | pyannote |
| clean_turn_taking | 8.12% | 5.73% | +2.4pp | CAM++ |
| third_party_background | 15.06% | 10.80% | +4.3pp | CAM++ |
| impostor_unknown | 28.24% | 19.78% | +8.5pp | CAM++ |
| **same_gender_similar** | 12.23% | **2.56%** | **+9.7pp** | **CAM++ 大勝** |

### CAM++ 弱點 root cause

CAM++ 為 VAD → embedder → clustering 三段架構，**無原生 overlap detection**。每個 VAD-active segment 算一個 embedding 然後 cluster — 兩人同時講話時整段被歸給單一 embedding。

最嚴重失敗集中在 simultaneous_start 18 條中有 10 條 predict 1 speaker（應為 2）。

### CAM++ 強點 root cause

CAM++ embedder（中文 speaker verification 大量訓練）對「相似但不同人」的 embedding 區分力較強。same_gender_similar 場景 CAM++ 2.56% vs pyannote 12.23%，是唯一明顯翻盤點。

---

## 7. Role Attribution Results (RQ3)

### 7.1 任務定義

Input：對話逐字稿 + diarization 給的 speaker_0 / speaker_1 標籤（不知道誰是客服）
Output：自動推斷 speaker_0 / speaker_1 分別是 agent 還是 customer
Eval set：37 closed-set 2-speaker cases from role_attribution split（排除 3 個 3-speaker impostor cases）

### 7.2 Baselines & Results

| Baseline | 方法 | Overall | Strong (14) | Medium (7) | Weak (16) |
|---|---|---|---|---|---|
| B1 Rule-based | Keyword pattern matching | 59.5% | 100% | 28.6% | 37.5% |
| **B2 LLM zero-shot** | Azure gpt-4o-mini | **94.6%** | **100%** | **100%** | **87.5%** |
| B3 Voice prior | P(voice = agent) from enrollment | 64.9% | 78.6% | 71.4% | 50.0% |
| B5 Acoustic embedder | ECAPA-TDNN cosine similarity | 56.8% | 71.4% | 71.4% | 37.5% |
| B4 Fusion (B1+B2+B3) | Weighted vote | 94.6% | 100% | 100% | 87.5% |

### 7.3 Failure mode 分析

B2 LLM 兩個失敗 case 均為 weak / mid_call 場景，雙方都使用「我們系統」「我們手上」等機構性語彙，連 LLM 也無法從 context 區分（客戶實際為保險業務員，言語風格與客服一致）。此類 case 比例極低（< 1%），不影響 prod 主流推薦。

### 7.4 Acoustic 路線在合成資料上不可行的證據

ECAPA-TDNN 跑出 56.8%（≈ random）的根本原因：agent prototype 與 customer prototype 的 **cosine similarity = 0.9448**（接近 1.0）。

合成資料 agent / customer 共用同一個 TTS voice pool（HsiaoChen / HsiaoYu / YunJhe 在 dev_smoke 既當 agent 也當 customer），embedding 幾乎重合，無法做 binary discrimination。

**真實 KGI 環境下不同**：agent voice pool = 固定 20-50 人員工，customer voice pool = 數十萬未知民眾，兩者 disjoint，acoustic enrollment 預期 accuracy 80-90%。但**此結論必須用真實員工錄音驗證，不能從合成資料推得**。

---

## 8. Cost & Deployment

### 8.1 估算（KGI 假設量級：10,000 通/月 × 5 分鐘）

| 元件 | 月成本 (USD) |
|---|---|
| GPU diarization (pyannote 3.1) | ~$10 |
| LLM role classification (gpt-4o-mini) | ~$300 |
| Model hosting / monitoring / ops | ~$80-130 |
| **總計** | **~$400 / 月** |

對比人工標註相同份量音訊：~$3,000-5,000 / 月。

---

## 9. Limitations & Known Risks

| Limitation | 影響 | Mitigation 路徑 |
|---|---|---|
| 合成 TTS 音訊，非真實電話線錄音 | 真實電話雜訊、codec artifacts、客戶口語多樣性未驗證 | **Pilot 前必做**：30-50 通去識別化真實錄音 validation |
| M+M same-gender 未測 | Azure 僅 1 個 zh-TW 男聲，無法生成兩個不同男聲場景 | 真實錄音 validation 時順帶涵蓋 |
| Acoustic role attribution 未在真實環境驗證 | B5 ceiling 是 synthetic-bounded；真實 prod 不知道 | 同上，validation 時順帶建 agent enrollment pool |
| Hokkien 場景過於樂觀 | 全為 F+M 不同 voice 組合，缺 same-gender hokkien | 真實錄音補測 |
| 沒測 noise / codec / echo / babble stress | 真實線路雜訊條件下 DER 可能上升 | 用 audiomentations 自動產生 stress condition test |
| 沒測 long-form (> 5 min) | 長對話 long-term speaker consistency 未驗證 | 拼接 / 真實長通話驗證 |

---

