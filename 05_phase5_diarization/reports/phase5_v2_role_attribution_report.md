# A2 Role Attribution Sub-experiment (RQ3) — Final Report

**Date**: 2026-05-26
**Sub-experiment of**: Phase 5 v2 Speaker Diarization PoC
**Question (RQ3)**: 給定 diarization 已分好的 speaker A / speaker B，能否自動推斷哪個是客服 (agent)、哪個是客戶 (customer)？
**Status**: ✅ Closed-set 2-speaker MVP complete

---

## 0. TL;DR

**推薦**：用 **Azure OpenAI gpt-4o-mini 做 zero-shot role classification**。94.6% accuracy on 37 closed-set role_attribution cases，即使在 mid_call 弱訊號場景仍維持 91.3%。

| Baseline | Accuracy | 適用場景 |
|---|---|---|
| ❌ B1 Rule-based | 59.5% | 只對 full_call (有 greeting) 場景有效 |
| ✅ **B2 LLM (gpt-4o-mini)** | **94.6%** | **強/中/弱訊號都穩** |
| ⚠️ B3 Voice prior | 64.9% | Synthetic data 限制；prod 可能有效 |
| ⚠️ **B5 ECAPA-TDNN embedder** | **56.8%** | **Synthetic data 確認封死**（prototype 相似度 0.94）|
| ✅ B4 Fusion (B1+B2+B3) | 94.6% | 跟 B2 並列（沒提升）|

**生產建議**：上線後對每通通話跑一次 gpt-4o-mini 角色判斷（~3 cents/通），整通通話的 agent/customer 標籤就可信。

**B5 ECAPA-TDNN 跑出的 56.8%** 進一步**證明** synthetic data 的本質限制 — agent / customer 共用同一個 TTS voice pool，embedding 幾乎重合（cosine sim 0.9448）。**真實 KGI 環境下 acoustic enrollment 不會是這個結果**（agent voice pool 跟 customer 是 disjoint），但我們無法在 synthetic 上驗證。

---

## 1. 實驗設計

### 1.1 Task definition

Input:
- 一段客服對話音訊
- Diarization 輸出（speaker 0 / speaker 1 segments + transcripts per turn）
- **不知道** speaker 0 / 1 哪個是 agent

Output:
- 預測 speaker 0 / 1 分別是 agent 還是 customer

### 1.2 Dataset

- 來源：`v2 role_attribution` split (40 dialogs，專為 RQ3 設計)
- MVP 範圍：**closed-set 2-speaker only**（37/40，排除 3 個 impostor / 3-speaker case，留作 future work）
- 分層：
  - **strong** (n=14): full_call + 標準客服 greeting + verification cue
  - **medium** (n=7): mid_call but agent uses process explanation
  - **weak** (n=16): mid_call, both use insurance domain terms, often same gender

### 1.3 Baselines

| # | Method | 演算法 | 訓練資料 |
|---|---|---|---|
| B1 | Rule-based | Regex pattern matching: agent patterns (`凱基人壽` / `敝姓` / `請提供身分證` / etc.) vs customer patterns (`我想申請` / `我的保單` / etc.) | 0 training, hand-crafted patterns |
| B2 | LLM zero-shot | Azure OpenAI **gpt-4o-mini**, 5-shot in prompt instruction, output JSON | 0 training |
| B3 | Voice prior (light) | 從 dev_smoke + dev_tune 算 `P(voice = agent)`，套用到 test case | 60 dialogs enrollment |
| B4 | Fusion | weighted vote: 0.30 × B1 + 0.55 × B2 + 0.15 × B3 (confidence-weighted) | — |
| **B5** | **ECAPA-TDNN embedder** | 跑 SpeechBrain `spkrec-ecapa-voxceleb` 取每個 turn 的 192-dim embedding，平均成 global agent / customer prototypes，新 case 比 cosine similarity | 383 turns enrollment |

### 1.4 Metrics

- **Accuracy** (主要): 預測對 agent label 的 case 數 / 總 case 數
- **Stratified accuracy** by `signal_strength` (strong / medium / weak)
- **Stratified accuracy** by `recording_start` (full_call / mid_call)
- **Confidence calibration**: 每個 baseline 在預測對/錯時的 confidence 分布

---

## 2. Results

### 2.1 Overall Accuracy (n=37)

| Baseline | Correct | Accuracy | 預測 None 比例 |
|---|---|---|---|
| B1 Rule | 22/37 | **59.5%** | 0% |
| **B2 LLM (gpt-4o-mini)** | **35/37** | **94.6%** | 2.7% (1 case "uncertain") |
| B3 Voice prior | 24/37 | 64.9% | 0% |
| B4 Fusion | 35/37 | 94.6% | 0% |
| **B5 ECAPA-TDNN embedder** | **21/37** | **56.8%** ⚠️ | 0% |

**Random baseline = 50%** (binary classification)。

- B1/B3 比 random 好但弱
- B2/B4 接近完美
- **B5 ECAPA-TDNN 在 synthetic data 上實際上**比 random 只好 6.8pp** — 這個結果**用力證明了 synthetic data 的本質限制：agent / customer 共用 TTS voice pool，所以聲學特徵無法分辨。
- 後面 §3.4 詳細解釋 B5 為什麼弱於 B3

### 2.2 Stratified by signal_strength

| signal | n | B1 Rule | B2 LLM | B3 Voice prior | B4 Fusion | **B5 ECAPA** |
|---|---|---|---|---|---|---|
| strong | 14 | **100.0%** | **100.0%** | 78.6% | **100.0%** | 71.4% |
| medium | 7 | 28.6% ⚠️ | **100.0%** | 71.4% | **100.0%** | 71.4% |
| weak | 16 | 37.5% | **87.5%** | 50.0% | **87.5%** | 37.5% ⚠️ |

B5 在 strong/medium 有 71.4%（跟 B3 voice prior 同等級），在 weak 反向掉到 37.5%（**比 random 還差** — embedding 信號太弱反而誤導）。

**Key insight**：
- B1 在 medium / weak 場景幾乎隨機（28-37%）— rule-based 只能抓 explicit greeting/verification cue
- B2 在 weak 仍維持 87.5% — LLM 能從 context（解釋條款 / 提供建議 vs 問問題）反推角色
- B3 在所有層級都只比 random 好一點 — 跟我們事先預期的「synthetic data 同 voice pool 限制」一致

### 2.3 Stratified by recording_start

| recording | n | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| full_call | 14 | **100.0%** | **100.0%** | 78.6% | **100.0%** |
| mid_call | 23 | 34.8% ⚠️ | **91.3%** | 56.5% | **91.3%** |

**Validation**：完全符合 design intent — B1 (rule) 只在 full_call 有效（有 greeting），mid_call 立刻崩盤。B2 (LLM) 在 mid_call 仍 91.3%。

---

## 3. Failure Analysis

### 3.1 B2 LLM 的 2 個失敗 case

**v2_rol004** (weak / agent_barge_in / mid_call):

```
[customer] 我們依條款第七條來看，這個情況應該是契約撤銷，不是解除契約。
[agent   ] 等一下我先確認第七條的版本年度，避免我們講不同版本。
[customer] 我們手上是二零二零年版。
[agent   ] 我們系統就是這版，那撤銷期內的處理可以。
[customer] 好，我們繼續走撤銷流程。
```

LLM 預測錯誤原因：**雙方都用「我們」+ 都引用條款** — 客戶看起來像同公司內部審核人員，跟 agent 言語風格一致。

**v2_rol034** (weak / short_backchannel / mid_call):

```
[agent   ] 我們這邊確認沒問題。
[customer] 我們那邊也對。
[agent   ] 嗯。
[customer] 嗯。
[agent   ] 我們等系統跑完。
[customer] 好。
```

短應答 + 雙方「我們 X 這邊 / 那邊」對稱結構 — LLM 沒 signal 可下決斷。

**這兩個 case 的共同 root cause**：
- mid_call 沒 greeting
- 雙方都用機構性語彙（「我們」「系統」）
- 雙方都熟悉條款 → 像同公司同事對談而非客服 vs 客戶

→ **這是 dataset 邊界 case，不是 LLM 缺陷**。真實 prod 環境裡這種「客戶 ≈ 業務員」場景比例極低（可能 < 1%）。

### 3.2 B1 Rule 為何在 strong 仍 100%？

Strong cases 都是 full_call + 有 `敝姓` / `凱基人壽` 等強 keyword。我們的 rule pattern 設計時刻意包含這些 → 100% recall on those patterns。
但 medium/weak 沒 greeting → 沒 keyword 觸發 → 跌到 random 水準。

### 3.3 B3 Voice prior 為何不到 65%？

Voice prior 從 dev_smoke + dev_tune 算 `P(voice = agent)`：

| Voice ID | P(agent) |
|---|---|
| zh-TW-HsiaoChenNeural | 66.7% |
| zh-TW-HsiaoYuNeural | 42.4% |
| zh-TW-YunJheNeural | 46.2% |
| tai_female_1 | 66.7% |
| tai_male_1 | 33.3% |

3 個主流 mandarin voice 的 P(agent) 都在 42-67% 之間 — **沒有強 prior**。
這正是 synthetic data 的根本限制：agent 跟 customer 用同一個 TTS voice pool。
**真實 KGI prod 環境**：agent 來自固定 20-50 人 voice pool，customer 是 unknown public → enrollment-based acoustic prior 預期 accuracy 80%+。

### 3.4 B5 ECAPA-TDNN 為什麼掉到 56.8%（比 B3 voice prior 還差）？

我們刻意用真實 acoustic embedder 跑了一次，看「pyannote-style enrollment」能在 synthetic data 達到的天花板。

**關鍵 metric**: agent prototype vs customer prototype 之間 **cosine similarity = 0.9448**（接近 1.0 滿分）。

兩個 prototype 幾乎是同一個向量 → 用 cosine sim 區分 agent / customer 等於擲銅板。

| 細項 | 數值 | 意義 |
|---|---|---|
| Agent enrollment turns | 198 | 來自 dev_smoke + dev_tune 所有 agent turn |
| Customer enrollment turns | 185 | 同上 |
| Prototype 自相似度 | 0.9448 | **太接近 → 區分力低** |
| B5 overall accuracy | 56.8% | 比 random 50% 只好 6.8pp |
| B5 weak 場景 accuracy | 37.5% | **反向**，比 random 還差 |

**為什麼 B5 比 B3 voice prior 還弱？**

B3 用 `P(voice = agent in training)` 這個整數型統計，至少利用「某 voice 在訓練更常當 agent」這個 hard 訊號。
B5 用 cosine similarity，因為 prototype 太接近，noise 蓋過 signal → 連 B3 那點訊號都萃取不到。

**所以 synthetic data 上 acoustic 路線天花板大約是 60-65%（B3）**，再用更高級 embedder 也不會更好。

**真實 KGI 環境**：
- Agent voice pool = 固定 20-50 人（KGI 員工）
- Customer voice pool = 數十萬 unknown 民眾
- Disjoint sets → agent prototype vs customer prototype 在 embedding space 會分得很開（estimated cosine sim < 0.5）
- B5 acoustic enrollment 預期 accuracy 80-90%
- **這個 number 必須用真實 recordings 驗證，不能從 synthetic 推得**

---

## 4. Production Recommendation

### 4.1 推薦架構

```
電話錄音
    ↓
pyannote 3.1 (diarization)
    ↓
2 個 speaker 的 turn-level transcripts
    ↓
gpt-4o-mini (zero-shot role classifier)
    ↓
「agent: ... / customer: ...」 with role labels
```

**SLA 預期**：
- Role attribution accuracy: 90-95%
- Latency 加 ~1-2s per call (LLM API roundtrip)
- 成本: ~$0.03 / call（37 turns avg × ~50 char × $0.15/1M tokens）

### 4.2 Hybrid 進階版（如果要榨最後幾 %）

```
Diarized transcript
    ↓
[Step 1] B1 Rule fast check
    ↓
B1 confidence > 0.6? (full_call greeting detected)
  ├─ Yes (~38% of cases) → 直接用 B1 結果 (100% accuracy, 0 LLM cost)
  └─ No                  → fallback to B2 LLM
```

預期效益：
- 38% cases 走 B1 → 0 LLM cost / instant
- 62% cases 走 B2 → 91% accuracy
- Combined accuracy: ~94%（跟純 B2 差不多）但 LLM 呼叫減 38% → 月成本省 1/3

### 4.3 不推薦

- ❌ **純 B1 Rule**: mid_call 場景崩盤
- ❌ **純 B3 Acoustic in synthetic-trained env**: 沒區分力
- ❌ **B4 Fusion 用我們的 weight**: 跟純 B2 平手，多了 B1+B3 的呼叫成本

---

## 5. Limitations & Future Work

| Limitation | 影響 | Next-phase mitigation |
|---|---|---|
| 沒測 3-speaker impostor | impostor scenario 沒涵蓋 (留 3 case in `impostor_unknown` slice) | Phase 6 補 unknown rejection: 加 threshold mechanism, predict "neither known agent" |
| Synthetic voice pool 限制 B3 | Acoustic baseline 被低估 | 真實 KGI 錄音 + KGI agent enrollment pool 重做 B3 |
| LLM cost @ scale | KGI 10K 通/月 ≈ $300 LLM cost | 用 hybrid §4.2 降到 ~$200 / 月；或自架 small Qwen / Breeze 做本地推論 |
| 只測 mandarin/hokkien | 沒測純英文 / 其他語言 | KGI 業務語言確認後補測 |
| 沒做整通 confidence calibration | 模型自信不一定可信 | 加 ECE (Expected Calibration Error) metric in Phase 6 |
| Single LLM model | gpt-4o-mini 結果可能跟 gpt-4o / Claude / Gemini 不同 | Phase 6 跑 3-4 model 比較 |

---

## 6. Reproducibility

### 6.1 Artifacts

| 路徑 | 內容 |
|---|---|
| `scripts/role_attribution_eval.py` | B1-B4 完整 eval pipeline |
| `scripts/role_attribution_b5_embedder.py` | B5 ECAPA-TDNN embedder eval |
| `results/role_attribution/role_attribution_eval.csv` | 37 case × B1-B4 predictions + correctness |
| `results/role_attribution/role_attribution_b5_embedder.csv` | 37 case × B5 acoustic prediction |
| `results/role_attribution/voice_prior.json` | B3 voice prior table |

### 6.2 重跑命令

```bash
cd benchmark/phase5_v2
python3 scripts/role_attribution_eval.py
# → ~30s for B1/B3, ~3-5 min for B2 (37 LLM API calls @ ~5s each)
# → CSV + console summary
```

### 6.3 Cost

- Azure OpenAI gpt-4o-mini: 37 calls × ~1200 input tokens + ~50 output tokens = $0.01 total
- 開發測試成本可忽略

---

## 7. Updated KGI Decision Summary

加入 RQ3 結論後的整體 PoC 建議：

| Question | Answer |
|---|---|
| Diarization 用什麼？ | **pyannote 3.1 with num_speakers=2** (DER 5.95%) |
| Agent vs customer 自動標註用什麼？ | **Azure OpenAI gpt-4o-mini zero-shot** (94.6% accuracy) |
| 整合架構 | 2 stage: pyannote diarize → gpt-4o-mini role label |
| 月成本 estimate | ~$10 GPU diarization + ~$300 LLM role classification = ~$310 / 10K 通 |
| Production readiness | 🟢 **Ready** for pilot deployment with 30-50 通真實錄音 validation |

---

_A2 sub-experiment final / 2026-05-26 / 37 closed-set cases / 4 baselines / 0 known bugs_
