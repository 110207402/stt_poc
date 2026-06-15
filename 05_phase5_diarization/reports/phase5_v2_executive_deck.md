# KGI Speech-to-Text PoC — Phase 5 Diarization + Role Attribution 結論

**對象**：KGI 高層 / 決策者 / Business stakeholder
**日期**：2026-05-26
**簡報時間預估**：10 分鐘
**對應技術文件**：`phase5_v2_final_report.md`

---

## 投影片 1 — 標題

# **採用 pyannote 3.1 + gpt-4o-mini，KGI 客服 STT 可上 production**

**Diarization Error Rate 5.95%**（產業 production-ready 標準：< 10%）
**Role attribution accuracy 94.6%**（自動分客服 vs 客戶）
**月運算成本 ~$400**（10,000 通客服電話）

---

## 投影片 2 — 3 個關鍵數字

| 維度 | 數值 | 業界 benchmark |
|---|---|---|
| **分人準確度 (DER)** | **5.95%** | 人工標註 inter-annotator 3-5%；模型已接近 |
| **角色判斷準確度** | **94.6%** | Random baseline 50% |
| **延遲 (latency)** | **< 3 秒** / 30 秒對話 | 即時 transcript 可行 |

**翻譯**：每通客服電話自動產生「客服講了什麼、客戶講了什麼、何時切換、誰是誰」，準確度接近人工，每通電話成本 ~$0.04。

---

## 投影片 3 — Production 部署架構

```
電話錄音 (mono, 16kHz)
    ↓
pyannote 3.1（語者分離）→ speaker_0 / speaker_1 時間軸
    ↓
ASR (Whisper 等)→ 逐字稿
    ↓
gpt-4o-mini（角色判斷）→ 哪個是客服、哪個是客戶
    ↓
最終 output：
  agent: 「您好凱基人壽⋯」
  customer: 「我想申請理賠⋯」
```

3 個元件、3 個現成 service、無需訓練自有模型、無需 fine-tune。

---

## 投影片 4 — 為什麼選 pyannote + gpt-4o-mini？

### Diarization：3 個模型比較（n=234 dialogs）

| 模型 | DER | 推薦？ |
|---|---|---|
| ✅ **pyannote 3.1** | **5.95%** | **採用** |
| 3D-Speaker CAM++ | 8.45% | 不採用（多 2.5pp）|
| pyannote 3.1 自動估說話人數 | 7.18% | 不採用（多 1pp）|

統計顯著性：pyannote 比 CAM++ 好的差距**不是隨機誤差**（p = 0.004, paired t-test）。

### Role attribution：5 種方法比較（n=37 dialogs）

| 方法 | Accuracy | 推薦？ |
|---|---|---|
| ✅ **Azure OpenAI gpt-4o-mini** | **94.6%** | **採用** |
| 規則 (keyword 比對) | 59.5% | 只在 full_call 有效（35% on mid_call） |
| 聲學 (ECAPA-TDNN embedder) | 56.8% | 合成資料封死，需真實員工錄音才能驗 |

---

## 投影片 5 — 成本估算

假設 KGI 每月 **10,000 通客服電話，平均 5 分鐘**：

| 項目 | 月成本 (USD) | 備註 |
|---|---|---|
| GPU diarization (pyannote 3.1) | ~$10 | 雲端 A100 GPU |
| LLM role classification (gpt-4o-mini) | ~$300 | Azure OpenAI |
| Model hosting + monitoring + ops | ~$80-130 | Inference service + alerting |
| **總計** | **~$400 / 月** | 對應 50,000 分鐘音訊處理 |

**對比**：人工標註相同份量音訊：~$3,000-5,000 / 月

**ROI**：每月省 $3,000+，1-2 個月回本。

**未來降本路徑**：上線 3-6 個月後累積真實員工錄音，可切到 hybrid 架構（rule + 聲學 enrollment + LLM fallback），月成本降至 ~$200。

---

## 投影片 6 — 已知 risk 與 mitigation

| Risk | 嚴重性 | Mitigation |
|---|---|---|
| 🔴 **沒測真實 KGI 客服錄音**（v2 全是 TTS 合成） | **HIGH** | **上 prod 前必做**：30-50 通去識別化錄音抽測，1-2 週 |
| 🟡 沒測電話線雜訊 / codec 干擾 | MEDIUM | 上線後監控、A/B 真實流量 vs lab 結果 |
| 🟡 罕見場景（同性別中年男對話、長通話 > 5 min） | MEDIUM | 上線後抽樣監控，必要時補測 |
| 🟢 多語言（mandarin / 台語 / codeswitch） | LOW | 已測，DER 都 < 8%（hokkien 1%, codeswitch 2%, mandarin 7%）|

---

## 投影片 7 — 13 種對話場景測試覆蓋

| 場景 | 模型表現 |
|---|---|
| 標準輪流講 / 短應答 / 長停頓 reentry | 全部 < 9% DER |
| Agent 插話 / Customer 打斷 / 兩人同時開口 | pyannote 3% 以下，CAM++ 弱（10-25%）|
| 純台語 / 中台夾雜 | 1-2% DER |
| 同性別相似聲線（最難） | pyannote 12% / CAM++ 2.6% — **CAM++ 唯一優勢場景** |
| 第三方背景 / 冒名查詢 (3 人通話) | 10-20% DER（罕見場景）|
| 純產品條款詢問（無 PII） | 0% DER |

234 通對話、1,514 個 turns、4 split、3 語言、13 slice。每個 slice 經 schema validation + voice diversity check + per-slice integrity check。

---

## 投影片 8 — 下一階段建議路線

### 短期（1-2 週）：上 prod 前必做
1. **真實 KGI 錄音 validation** — 30-50 通去識別化錄音 → 跑 pyannote + gpt-4o-mini → 確認 DER 跟 role accuracy 在真實場景仍維持
2. 同步收集 KGI 員工 voice 樣本，為未來 acoustic enrollment 鋪路

### 中期（1 個月）：完整 STT pipeline 整合
3. **ASR 整合** — Whisper / Breeze-ASR 跟 pyannote / gpt-4o-mini 整合，產生 with-speaker-tag 逐字稿
4. 端到端 latency / cost 量化

### 長期（3-6 個月）：production scale-up
5. **Pilot A/B test** — KGI 現有系統 vs 我們推薦 1 個月 production traffic 比對
6. **Hybrid 降本架構** — 建 acoustic enrollment pool 切換降低 LLM 成本
7. **Stress test** — 真實噪音 / 不同電話線品質 / 長通話 (> 5 min)

---

## 投影片 9 — 需要 KGI 決策的事

| Decision | Owner | 為什麼需要 |
|---|---|---|
| ✅ **批准採用 pyannote 3.1 + gpt-4o-mini pipeline** | 業務 + IT 高層 | PoC 結論的核心 deliverable |
| ✅ **提供 30-50 通去識別化客服錄音** | Legal + IT | 完成 risk mitigation |
| ⏳ **核可 Phase 6 budget**（ASR 整合 + pilot） | Budget owner | 約 1 個月 engineer time |
| ⏳ **指定 KGI infra team 對接窗口** | IT director | 後續 deployment 對接 |

---

## 投影片 10 — Summary（給沒空看細節的人）

> **Phase 5 PoC 結論：用 pyannote 3.1 做語者分離 + gpt-4o-mini 做角色判斷。**
>
> - 分人準確度 5.95%（接近人工標註）
> - 角色判斷準確度 94.6%
> - 月成本 ~$400（10K 通電話），ROI 1-2 個月
>
> **但這基於 234 通 TTS 合成對話。上 prod 前需要 30-50 通真實 KGI 客服錄音來最後驗證。**
>
> 驗證通過 → 可開始 production pilot
> 驗證失敗 → 已知 fallback 方案（hybrid pyannote + CAM++），不會 surprise

---

## 附錄：詳細文件位置

| 文件 | 路徑 |
|---|---|
| 完整技術 report (12 sections) | `benchmark/phase5_v2/reports/phase5_v2_final_report.md` |
| Role attribution 子實驗 report | `benchmark/phase5_v2/reports/phase5_v2_role_attribution_report.md` |
| Per-case raw 數據 | `benchmark/phase5_v2/results/` |
| 234 對話原始 spec | `benchmark/phase5_v2/data/dialog_specs/all_dialogs.json` |
| 234 對話音訊 + ground truth | `benchmark/phase5_v2/audio/` + `benchmark/phase5_v2/data/ground_truth_per_case/` |

---

_Phase 5 v2 executive deck v2.0 / 2026-05-26 / for KGI Life Insurance decision-makers_
