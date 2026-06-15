# KGI 人壽 STT 系統概念驗證 — 完整交接檔案包

> 凱基人壽客服語音轉文字（STT）五階段概念驗證的完整交接包：簡報，以及五個階段全部的**報告 / 資料 / 腳本 / 原始結果**。

對應簡報：[`presentation/KGI_STT_PoC.pptx`](presentation/KGI_STT_PoC.pptx)（43 張，含內嵌音檔）

依簡報五個階段對應 `01`–`05` 五個子資料夾，每個子資料夾含當階段的 **報告 / 資料 / 腳本 / 結果**。無重複舊版檔案。

---

## 先看這裡：簡報

| 檔案 | 說明 |
|---|---|
| [`presentation/KGI_STT_PoC.pptx`](presentation/KGI_STT_PoC.pptx) | **最終簡報 43 張**。1–35 為主線（五階段＋整合＋致謝），36–43 為附錄。多頁內嵌原始通話音檔，可現場點圖示播放。 |
| [`presentation/KGI_STT_PoC.pdf`](presentation/KGI_STT_PoC.pdf) | 簡報的 **PDF 視覺版**（無 PowerPoint／字型也能看；音檔僅存在於 .pptx）。 |

---

## 資料夾結構

```
KGI_STT_PoC_Bundle/
├─ README.md                                  ← 本檔案
│
├─ presentation/                              ← 交接核心：簡報
│   ├─ KGI_STT_PoC.pptx                       ← 最終簡報（43 張，內嵌音檔）
│   └─ KGI_STT_PoC.pdf                        ← 簡報 PDF 視覺版
│
├─ 01_streaming_benchmark/                    ← 簡報 §01（Slides 4–8）
│   ├─ reports/
│   │   ├─ analysis_report.md                 ← Phase 1 完整深度分析
│   │   ├─ comparison.md
│   │   ├─ model_summary.csv                  ← 6 模型彙整 metrics
│   │   ├─ run_metrics.csv                    ← 1,800 runs 原始資料
│   │   ├─ streaming_models_comparison.md     ← 給高階看的精簡版
│   │   └─ streaming_models_comparison.pdf
│   ├─ data/
│   │   └─ cases_seed.csv                     ← 50 條凱基客服腳本
│   ├─ audio/
│   │   └─ d0001.wav … d0010.wav             ← 10 條中文客服輸入音檔範例（含簡報用 d0001 / d0010）
│   └─ scripts/
│       ├─ run.py                             ← 串流測試 runner
│       ├─ streamer.py                        ← WebSocket 串流客戶端
│       ├─ noise.py                           ← 噪音注入（white/pink）
│       ├─ metrics.py                         ← CER / TTFP / RTF 計算
│       ├─ analyze.py                         ← 結果分析
│       ├─ report.py                          ← 報告生成
│       └─ tts.py                             ← Yating TTS 音檔生成
│
├─ 02_phase1_hokkien_batch/                   ← 簡報 §02（Slides 9–13）
│   ├─ reports/
│   │   ├─ phase2_analysis_report.md          ← 完整 Phase 2 分析
│   │   ├─ phase2_batch/batch_metrics.csv     ← zipformer 結果
│   │   ├─ phase2_batch_qwen/batch_metrics.csv
│   │   ├─ phase2_batch_funasr/batch_metrics.csv
│   │   └─ phase2_batch_firered/batch_metrics.csv
│   ├─ data/
│   │   ├─ hard_cases.csv                     ← 20 條最難中文 case
│   │   └─ hokkien_cases.csv                  ← 20 條台語 case（含 h0001 / h0018 GT）
│   ├─ audio/
│   │   └─ h0001.wav … h0010.wav             ← 10 條台語客服輸入音檔範例（含簡報用 h0001）
│   └─ scripts/
│       ├─ eval_batch.py                      ← 批次模型 runner
│       ├─ generate_hokkien_tts.py
│       └─ tts_hokkien.py                     ← facebook/mms-tts-nan 備案
│
├─ 03_phase2_breeze_deep/                     ← 簡報 §03（Slides 14–18）
│   ├─ reports/
│   │   └─ breeze_asr_26_report.md            ← Breeze-ASR-26 深度評估完整報告（含逐 case HYP/REF）
│   └─ scripts/
│       ├─ run_breeze_phase3_deep.ipynb       ← Colab 推論 notebook（Breeze BF16，160 runs）
│       ├─ noise.py                           ← 噪音注入 codec/echo/babble（與 Phase 1 共用）
│       ├─ metrics.py                         ← CER 計算 OpenCC s2tw（與 Phase 1 共用）
│       └─ README.md                          ← 本階段 scripts 說明
│
├─ 04_breeze_pii_pipeline/                    ← 簡報 §04（Slides 19–25）
│   ├─ data/
│   │   ├─ phase3_queries.csv                 ← 90 條 query（含 digit-spaced 寫法修正）
│   │   └─ .env.example                       ← Azure API 設定範本
│   ├─ scripts/
│   │   ├─ generate_phase3_tts.py             ← Yating TTS 90 條音檔生成
│   │   ├─ run_breeze_asr_phase3.ipynb        ← Colab 推論 notebook（E1/E2）
│   │   ├─ eval_asr_results.py                ← ASR 評估（CER + PII recall）
│   │   ├─ run_pii_eval.py                    ← PII 評估（M1 ONNX + M2 Azure）
│   │   └─ azure_client.py                    ← Azure chat client helper
│   ├─ results/
│   │   ├─ phase3_breeze_asr_results.csv      ← Breeze hyp_e1 / hyp_e2 原始輸出
│   │   ├─ phase3_asr_eval_report.md
│   │   ├─ phase3_asr_eval_per_query.csv
│   │   ├─ phase3_asr_eval_summary.csv
│   │   ├─ phase3_pii_eval_report.md
│   │   ├─ phase3_pii_eval_per_query.csv      ← PII 逐 query level_* 命中（簡報 39 頁來源）
│   │   ├─ phase3_pii_eval_summary.csv
│   │   └─ phase3_pii_detections.json         ← M1/M2 偵測 cache
│   ├─ reports/
│   │   └─ phase3_final_report.md             ← Phase 3 整合分析報告
│   └─ audio/
│       ├─ manifest.json                      ← 音檔 metadata（全 90 條）
│       └─ q001.wav … q010.wav                ← 10 條 TTS 音檔範例（全 90 條 query 見 data/phase3_queries.csv）
│
└─ 05_phase5_diarization/                     ← 簡報 §05（Slides 26–33）
    ├─ reports/
    │   ├─ phase5_v2_final_report.md          ← 完整技術 report（12 sections）
    │   ├─ phase5_v2_role_attribution_report.md ← RQ3 子實驗 report（B1–B5）
    │   ├─ phase5_v2_executive_deck.md        ← 高層 10-slide deck
    │   ├─ dialog_cleanup_changelog.md        ← Dataset cleanup 變更紀錄
    │   └─ v1_vs_v2_design_diff.md            ← v1 vs v2 設計差異
    ├─ design/
    │   ├─ phase5_v2_experiment_design.md     ← 完整實驗設計（RQ1/2/3）
    │   └─ dialog_spec.schema.json            ← JSON Schema 驗證規格
    ├─ data/
    │   ├─ dialog_specs/all_dialogs.json      ← 234 dialog 完整 spec
    │   └─ ground_truth_per_case/
    │       ├─ speech/                        ← 234 VAD-trimmed RTTM
    │       └─ turn/                          ← 234 turn-ownership RTTM
    ├─ audio/
    │   └─ dialogs_clean/                     ← 10 條對話音檔範例（16kHz mono；GT 仍為全 234 條）
    ├─ scripts/
    │   ├─ generate_audio_and_gt.py           ← TTS + VAD + RTTM pipeline
    │   ├─ phase5_v2_models_only.ipynb        ← Colab notebook（3 diarization models）
    │   ├─ role_attribution_eval.py           ← RQ3 B1–B4 evaluation
    │   └─ role_attribution_b5_embedder.py    ← RQ3 B5 ECAPA-TDNN evaluation
    └─ results/
        ├─ annotated_transcripts_pyannote_31_fixedN.csv ← ★ 最終交付物：3,814 段逐段標註逐字稿
        ├─ metrics_pyannote_31_auto.csv       ← Diarization per-case（234 × 3 model）
        ├─ metrics_pyannote_31_fixedN.csv
        ├─ metrics_campp_3dspeaker.csv
        ├─ cross_model_summary.csv
        ├─ der_by_slice.csv                   ← 13 slice 細分（簡報 40 頁來源）
        ├─ der_by_language.csv                ← 3 語言細分
        └─ detections/                        ← per-case 預測 JSON
```

---

## 簡報投影片 ↔ 資料夾對應（最終 43 張）

| 投影片 | 章節 | 對應資料夾 | 重點 |
|---|---|---|---|
| 1–3 | 開場 / 核心問題 | — | 業務問題與四個子問題骨架 |
| **4–8** | §01 串流 ASR | `01_streaming_benchmark/` | 6 模型 × 50 cases × 3 noise × 2 = 1,800 runs；zipformer-zh-xl 冠軍（CER 10.70%）。第 8 頁含 CER／首字延遲／**RTF**／關鍵差異 |
| **9–13** | §02 台語盤點 | `02_phase1_hokkien_batch/` | 加入 3 批次模型；除 Breeze 外台語全軍覆沒（47–54% CER） |
| **14–18** | §03 Breeze 深評 | `03_phase2_breeze_deep/` | Breeze-ASR-26：Hokkien 21.78% vs 官方 30.13%，領先其他三家 25–32pp |
| **19–25** | §04 PII pipeline | `04_breeze_pii_pipeline/` | Breeze + Azure GPT-4o-mini 端到端；M2 PII recall **94.81%** |
| **26–33** | §05 語者分離 + 角色 | `05_phase5_diarization/` | pyannote 3.1 DER **5.95%**（顯著贏 CAM++ p=0.004）；gpt-4o-mini role **94.6%** |
| 34–35 | 整合 / 致謝 | 全部 | end-to-end pipeline；月成本 ~$400；上 prod 前須 30–50 通真實錄音驗證 |
| **36–43** | 附錄 | 各階段 | 37 弱點分析、38 失敗案例 h0012、39 PII 逐類別、40 Phase 5 完整 13 場景、**41–43 三例（d0001／h0001／h0018）完整逐字轉錄** |

> 附錄 41–43 的完整轉錄逐字內容，原始來源分別為 `01_.../reports/run_metrics.csv`（d0001）、`02_.../reports/phase2_batch_*`（h0001）、`03_.../reports/breeze_asr_26_report.md`（h0018）。

---

## 重現步驟（各階段）

> 每個資料夾都自帶 10 條輸入音檔範例，可直接驗證流程；要完整重跑（全資料集）需補上對應的 API key／模型權重／Colab GPU，各步驟已標註。

### Phase 1 · `01_streaming_benchmark`（中文串流 ASR · 1,800 runs）
需要：sherpa-onnx 串流伺服器（`server.py`）＋ 6 個串流模型權重（原始 `benchmark/` repo／HuggingFace）、Yating TTS key。
```bash
cd 01_streaming_benchmark
python scripts/tts.py                       # 1) 生成中文客服 TTS（已含 10 條於 audio/）
python scripts/run.py \
  --models zipformer-zh-xl,zipformer-zh-sm,paraformer,paraformer-tri,zipformer-zh-xl-t,zipformer-zh-sm-t \
  --noise-types clean,white,pink --repeats 2   # 2) 起本機 sherpa server 並串流 → reports/run_metrics.csv
python scripts/analyze.py                   # 3) 分析 → reports/.../analysis_report.md
python scripts/report.py                    # 4) 高階對照 → streaming_models_comparison.md
```
> `run.py` 以 `--host 127.0.0.1 --base-port 18000` 為每個模型起一個 sherpa-onnx server 子程序；`server.py` 與模型權重不在本包內。

### Phase 2 · `02_phase1_hokkien_batch`（台語批次盤點 · 480 runs）
需要：Yating TTS key（台語音檔）、sherpa-onnx（FunASR-Nano／fire-red-asr2）、Qwen3-ASR API。
```bash
cd 02_phase1_hokkien_batch
python scripts/generate_hokkien_tts.py      # 1) 生成台語 TTS（已含 10 條於 audio/）
python scripts/eval_batch.py \
  --models fun-asr-nano,fire-red-asr2,qwen3-asr \
  --noise-types clean,codec,echo,babble --snr 15   # 2) 批次推論 → reports/phase2_batch*/batch_metrics.csv
```

### Phase 3 · `03_phase2_breeze_deep`（Breeze 深評 · 160 runs）
需要：Colab GPU（A100）；Breeze-ASR-26 由 HuggingFace 自動下載。`noise.py`／`metrics.py` 為共用函式庫，由 notebook import。
```text
1. 把整個 bundle 上傳到雲端硬碟根目錄
2. Colab 開 scripts/run_breeze_phase3_deep.ipynb（執行階段選 GPU），由上而下執行：
   載入 Breeze（BF16）→ 中文20+台語20 注入 clean/codec/echo/babble → 30s 分塊 greedy → CER
3. 輸出 results/phase3_breeze_deep_results.csv；逐 case 分析見 reports/breeze_asr_26_report.md
```

### Phase 4 · `04_breeze_pii_pipeline`（Breeze + PII 端到端 · 360 評估點）
需要：Azure GPT-4o-mini deployment（M2）、onnxruntime（M1 OpenAI Privacy Filter）、Yating TTS key。
```bash
cd 04_breeze_pii_pipeline
cp data/.env.example .env                   # 填入 AZURE_API_KEY 等
python scripts/generate_phase3_tts.py       # 1) 生成 90 條音檔（已含 10 條於 audio/）
# 2) 在 Colab 開 scripts/run_breeze_asr_phase3.ipynb 跑 ASR → results/phase3_breeze_asr_results.csv
python scripts/eval_asr_results.py          # 3) ASR 評估（CER + 術語召回）
python scripts/run_pii_eval.py --methods m1,m2   # 4) PII 評估（M1 ONNX + M2 Azure）
```

### Phase 5 · `05_phase5_diarization`（語者分離 + 角色 · 234 通）
需要：HF token（pyannote 3.1 為 gated 模型）、Azure Speech＋Yating（生對話音檔）、Azure GPT-4o-mini（角色判斷）。
```bash
cd 05_phase5_diarization
python scripts/merge_dialog_specs.py        # 1) 合併 + 驗證 dialog spec
python scripts/validate_against_schema.py   #    schema 嚴格驗證（0 error 才 freeze）
python scripts/generate_audio_and_gt.py     # 2) TTS+VAD → audio/ + GT rttm（已含 10 條 + 全 234 GT）
# 3) 在 Colab 開 scripts/phase5_v2_models_only.ipynb 跑 3 個語者分離模型
#    → results/metrics_*.csv + results/detections/
python scripts/role_attribution_eval.py         # 4) RQ3 角色 B1–B4 → results/role_attribution/role_attribution_eval.csv
python scripts/role_attribution_b5_embedder.py  #    RQ3 角色 B5（聲紋）→ results/role_attribution/role_attribution_b5_embedder.csv
```

---

## 關鍵數字速查

- **Phase 1 串流模型** — zipformer-zh-xl，乾淨 CER 10.70%、首字延遲 728ms、RTF ≈ 1.04
- **Phase 2 台語盤點** — 3 批次模型對台語全軍覆沒（47–54% CER）
- **Phase 3 Breeze 深評** — Mandarin 6.54% / Hokkien 21.78%，領先其他模型 25–32pp
- **Phase 4 端到端 PII** — Breeze（CER_norm 13.54%）+ Azure GPT-4o-mini（PII recall **94.81%**）
- **Phase 5 語者分離 + 角色** — pyannote 3.1 DER **5.95%**（顯著贏 CAM++ p=0.004）、gpt-4o-mini role **94.6%**、月成本 ~$400

---

## 環境依賴

- Python 3.12；transformers 4.57+、openai 1.10+、python-dotenv
- PII：onnxruntime（M1 ONNX path）、Azure OpenAI deployment（M2）
- ASR：Breeze-ASR-26（HuggingFace，首次自動下載）、Google Colab A100（推論）
- 語者分離：pyannote.audio 3.1（gated，需 HF token 並到模型頁按 Agree）
- TTS：Yating TTS API key（如需重生音檔）

---


> 註：簡報內嵌的原始通話音檔（d0001 / h0001 / h0018 / q001 / v2_rol001 等）已**直接打包在 `.pptx` 內**，離線也能播放。
