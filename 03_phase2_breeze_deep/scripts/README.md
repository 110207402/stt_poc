# Phase 3 · Breeze 深評 — scripts

Phase 3（Breeze-ASR-26 深度評測，20 難中文 + 20 台語 × 4 噪音 = 160 runs）的程式：

| 檔案 | 用途 |
|---|---|
| `run_breeze_phase3_deep.ipynb` | Colab（A100 / BF16）推論主程式：載入 Breeze-ASR-26 → 對中文 20＋台語 20 注入 clean/codec/echo/babble → 30s 分塊 greedy 推論 → 算 CER |
| `noise.py` | 噪音注入 clean / codec(G.711 μ-law) / echo(room reverb) / babble(4 人 @ SNR 15dB)（與 Phase 1 共用） |
| `metrics.py` | CER：OpenCC `s2tw` 繁化 → 去標點與空白 → Levenshtein（與 Phase 1 共用） |

## 重現

1. 把整個 `KGI_STT_PoC_Bundle/` 上傳到雲端硬碟根目錄。
2. Colab 開 `run_breeze_phase3_deep.ipynb`，執行階段選 **GPU（A100）**，由上而下執行。
3. 輸出 `../results/phase3_breeze_deep_results.csv`；逐 case HYP/REF 與每種噪音的 CER 分析見 `../reports/breeze_asr_26_report.md`。

> 推論設定：BF16、30 秒手動分塊（最後一塊 < 0.5s 併入前塊）、`num_beams=1` greedy、噪音 SNR 15dB —— 方法完整記錄於報告 §2。
> bundle 內每個音檔資料夾只放 10 條範例；notebook 會自動只跑存在的音檔，指向完整資料集即可重現全部 160 runs。
