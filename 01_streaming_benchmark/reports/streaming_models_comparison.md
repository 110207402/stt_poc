# 串流 ASR 模型比較報告

## 1. 測試音檔

| 項目 | 內容 |
|---|---|
| 案例數 | 50 條 TTS 生成 |
| 來源 | 凱基人壽客服情境腳本 — 理賠申請／保單變更／商品詢問／保單查詢／失效復效等 |
| 語言 | 繁體中文（國語） |
| 格式 | 16 kHz mono WAV |
| 噪音條件 | clean / white 20dB / pink 20dB（3 種）|
| 重複次數 | 每組合 2 次 |
| **總 runs** | **6 模型 × 50 cases × 3 noise × 2 repeats = 1800** |

## 2. 受測模型

| 代號 | 來源 | 框架 |
|---|---|---|
| zipformer-zh-xl | sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30 | sherpa-onnx |
| zipformer-zh-xl-t | sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30 | sherpa-onnx |
| zipformer-zh-sm | sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30 | sherpa-onnx |
| zipformer-zh-sm-t | sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01 | sherpa-onnx |
| paraformer | sherpa-onnx-streaming-paraformer-bilingual-zh-en | sherpa-onnx |
| paraformer-tri | sherpa-onnx-streaming-paraformer-trilingual-zh-cantonese-en | sherpa-onnx |

## 3. 實驗方式

- 每條音檔以 WebSocket chunk 串流方式餵入模型
- 紀錄逐字 partial / final 輸出與時間戳
- CER 計算：去標點空白後 Levenshtein / |ref|
- 噪音以指定 SNR 程序混入後重跑相同流程

**指標定義**

#### 準確度類

| 指標 | 全名 | 計算方式 | 解讀 |
|---|---|---|---|
| **CER** | Character Error Rate（字元錯誤率）| Levenshtein(ref, hyp) ÷ \|ref\| | 把參考答案和模型輸出都去掉標點與空白，計算最少需要幾次「插入／刪除／替換」單字編輯才能把 hyp 變成 ref，再除以 ref 字數。CER 越低越好；一般 < 5% 為優、5–10% 可用、> 15% 偏差。|
| **CER mean** | 平均 CER | 所有 case 的 CER 取算術平均 | 反映整體平均表現，但容易被少數極差 case 拉高。 |
| **CER p50** | CER 中位數 | 把所有 case CER 排序後取正中間值 | 反映「典型 case」的表現；不受極端 case 影響。若 mean 比 p50 高很多 → 代表分布偏斜，有災難級 case。 |
| **CER min / max** | 最佳／最差 case CER | 50 條 case 中表現最好／最差那條 | 用來看模型表現的下限與上限分布範圍。 |

#### 延遲類（串流模型才有意義）

> **Partial vs Final 概念**：串流 ASR 在使用者說話過程中會不斷吐出**暫定的轉錄結果（partial）**，等到偵測句子結束或信心足夠後，才把該段確定下來變成 **final**。partial 會持續被覆寫，final 不會被改。

| 指標 | 全名 | 計算方式 | 解讀 |
|---|---|---|---|
| **TTFP** | Time to First Partial | 音檔開始送進模型 → 第一個 partial 文字出現的毫秒數 | **使用者感受到的反應速度** — TTFP 越低，UI 上字幕越快出現。對話式應用建議 < 1000 ms；> 1500 ms 會感覺「卡住」。 |
| **TTFF** | Time to First Final | 音檔開始 → 第一個 final 出現的毫秒數 | 第一段話被「正式確認」要多久。TTFF 通常比 TTFP 高很多（要等句子結束 + endpoint 偵測）。 |
| **E2E** | End-to-End latency | 音檔開始 → 最後一個 final 出現的總毫秒數 | 整段轉錄完成的總時間。對 ~40 秒的音檔，E2E ≈ 42 秒表示「邊聽邊吐字、聽完隨即收尾」。 |
| **RTF** | Real-Time Factor | E2E ÷ 音檔長度 | 處理速度的相對指標。**RTF < 1.0** → 比即時更快（聽 1 秒處理 < 1 秒），可用於即時串流。RTF = 1.0 → 剛好即時。RTF > 1.0 → 比即時還慢，串流會持續積壓。 |

## 4. 整體準確度排名（clean）

| 排名 | 模型 | CER mean | CER p50 | CER min | CER max |
|---|---|---|---|---|---|
| 1 | **zipformer-zh-xl** | **10.70%** | 10.19% | 7.14% | 16.57% |
| 2 | zipformer-zh-xl-t | 10.93% | 10.57% | 6.25% | 17.68% |
| 3 | paraformer | 11.17% | 10.28% | 6.49% | 16.38% |
| 4 | zipformer-zh-sm-t | 11.83% | 11.44% | 7.79% | 17.13% |
| 5 | zipformer-zh-sm | 13.28% | 12.83% | 7.78% | 18.47% |
| 6 | paraformer-tri | 14.02% | 13.31% | 8.38% | 21.12% |

## 5. 抗噪比較

| 模型 | clean | white 20dB | pink 20dB | 最差漲幅 |
|---|---|---|---|---|
| zipformer-zh-xl | 10.70% | 11.32% | 11.11% | +0.62% |
| zipformer-zh-xl-t | 10.93% | 12.57% | 12.10% | +1.64% |
| **paraformer** | 11.17% | 11.32% | 11.13% | **+0.15%** |
| zipformer-zh-sm-t | 11.83% | 13.48% | 13.40% | +1.65% |
| zipformer-zh-sm | 13.28% | 13.78% | 13.36% | +0.50% |
| paraformer-tri | 14.02% | 14.18% | 13.73% | +0.16% |

## 6. 延遲比較（clean）

| 模型 | TTFP (ms) | TTFF (ms) | E2E (ms) | RTF |
|---|---|---|---|---|
| **zipformer-zh-sm** | **681** | 8381 | 41760 | 1.032 |
| zipformer-zh-xl | 728 | 8615 | 42023 | 1.038 |
| zipformer-zh-sm-t | 763 | 8517 | 41921 | 1.036 |
| zipformer-zh-xl-t | 800 | 8559 | 42100 | 1.040 |
| paraformer | 932 | 8500 | 41841 | 1.034 |
| paraformer-tri | 934 | 8495 | 41898 | 1.035 |

## 7. 各維度冠軍

| 維度 | 冠軍 | 數字 |
|---|---|---|
| 最高準確度 | **zipformer-zh-xl** | clean CER 10.70% |
| 最強抗噪 | paraformer | white 漲幅 +0.15% |
| 最快首字反應（TTFP） | zipformer-zh-sm | 681 ms |
