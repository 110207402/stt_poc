# Phase 5 v1 vs v2 — 設計變更總覽

**目的**：說明 v2 改了哪裡、為什麼改、v1 結果在 v2 框架下怎麼解讀

---

## 改動清單

### A. Ground Truth（最關鍵變更）

| 維度 | v1 | v2 |
|---|---|---|
| GT 來源 | TTS turn 完整時長 → RTTM | TTS turn → **VAD 偵測實際發聲** → trim silence → RTTM |
| GT 檔案數 | 1（`ground_truth.rttm`）| **3**（speech / turn / overlap_regions）|
| GT 含 silence | ✗（silence 被當 speech）| ✓（只標真實 speech）|
| 量化問題 | GT 區間僅 65.8% frame 真有聲 | VAD active ratio 預期 0.6-0.85 健康區間 |
| 對 DER 影響 | pyannote/Sortformer 被高估 10-15pp | 預期 pyannote DER 從 29% → 12-18% |

### B. pyannote 設定

| 維度 | v1 | v2 |
|---|---|---|
| Speaker count | automatic | **必跑兩個 config：`auto` + `num_speakers=2`** |
| Speaker count accuracy | 75.6% | 預期 fixed2 配置 100% |
| 預期 DER 改善 | - | 額外 3-5pp |

### C. Dataset 架構

| 維度 | v1 | v2 |
|---|---|---|
| Split | 沒分 | **dev_smoke (20) / dev_tune (30) / test_core (120) / test_stress (60) / role_attribution (40)** |
| Slice 類型 | 7 (T1-T7) | **12**（補 customer_barge_in、simultaneous_start、same_gender_similar、third_party_background、short_backchannel）|
| Overlap 總時長 | 41 秒（1.3%）| 預期 > 5 分鐘 |
| Stress 變體 | 沒有 | **60**（codec / echo / babble / low_volume）|
| 三方通話 | 沒有 | **6** dialogs |
| 同性別 stress | 沒有 | **10** dialogs |

### D. Dialog Spec

| 維度 | v1 | v2 |
|---|---|---|
| 格式 | CSV + JSON 塞 cell | **JSON Schema-validated**（每 case 一份 spec） |
| 元資料 | 基本 | persona / role_signal_strength / recording_start / tags per turn |
| Voice 指定 | 隨機 pool | 明確 voice_id |
| 可重現性 | random.seed | 寫死 spec，永遠一樣 |

### E. Model Matrix

| 維度 | v1 | v2 |
|---|---|---|
| Pyannote configs | 1（auto only） | **2 × 2 = 4**（3.1/community-1 × auto/fixed2） |
| Sortformer | v1 offline | v1 offline + (stretch) v2 |
| CAM++ | ✓ | ✓ |
| ERes2NetV2 | 不確定有沒有真的跑（sv_model silent no-op） | **patched，含 sanity-check delta verification** |
| SOND | 跑不動，浪費時間 | **跳過**，文件記錄 "library issue, deferred" |
| DiariZen | 沒考慮 | (stretch) 列入但只當 research reference（CC BY-NC 4.0 不可商用） |
| 總 model configs | 4 | **7 必跑 + 3 stretch** |

### F. Metrics

| 維度 | v1 | v2 |
|---|---|---|
| DER 變體 | strict + lenient | strict + lenient + **no_overlap + overlap_only** |
| Consistency | speaker_count_correct, purity | + **fragmentation_rate**, **label_swap_count**, **same_speaker_link_accuracy** |
| Overlap | overlap_ratio | **detection_recall + precision + F1 + two_speaker_active_ratio** |
| Role attribution | 沒做 | **5 個專屬 metrics**（per role precision/recall + error type） |
| Operational | RTF | RTF + **peak_GPU_mem + model_size + load_time** |
| **總 metrics 數** | ~5 | **21** |

### G. Role Attribution（RQ3）

| 維度 | v1 | v2 |
|---|---|---|
| 設計 | 沒做（屬於 Phase 5 後續）| **獨立子實驗**，40 dialogs |
| Baseline 數 | 0 | **4**（rule / text / acoustic / fusion） |
| Enrollment pool | 沒有 | **8-12 個 agent voice samples** |
| Unknown rejection | 沒測 | impostor 集設計專測 |

### H. Reproducibility

| 維度 | v1 | v2 |
|---|---|---|
| Model run metadata | 部分（CSV 只有 model name + DER） | **每個 model run 一份 model_metadata.json**（含 pkg 版本、GPU、command、license、tuned-on-split） |
| Random seed | 部分 | **每個 stage 明確 seed** |
| Git tagging | 沒做 | **audio frozen at git tag**，model run 開跑後不准改 |
| Notebook state risk | 高（in-memory）| 低（結果寫死到檔案，下個 notebook 讀檔）|

### I. Execution Plan

| 維度 | v1 | v2 |
|---|---|---|
| Stages | 隨意 | **6 個明確 stage with explicit gates** |
| Gate criteria | 無 | 每個 stage 通過條件清楚（VAD 健康、DER 範圍、樣本完整）|
| 失敗動作 | 摸黑 debug | 明確 fallback（dev_smoke 不通就回去調 VAD） |

---

## v1 結果在 v2 框架下的可信度

### 可保留的結論

1. ✓ **CAM++ 比 pyannote 強很多** — v1 顯示 12% vs 29%，差距太大不太可能因 GT 修正而完全反轉
2. ✓ **Sortformer 對 Hokkien 失敗** — Hokkien 47% DER 不是 GT 問題，是 Sortformer 訓練資料偏英文
3. ✓ **整體 model ranking**（CAM++ > Sortformer > pyannote）大致正確
4. ✓ **CAM++ 在 stress condition 下的 robustness** 還沒測，但 baseline 表現好

### 不可保留的數字

1. ✗ **絕對 DER 數值** — v1 GT 不準，所有絕對數字要重算
2. ✗ **pyannote vs Sortformer 誰好** — v1 差距 < 2pp，GT 修了可能反轉
3. ✗ **ERes2NetV2 跑出來跟 CAM++ 一模一樣** — patch 沒驗證真生效，結論不可信
4. ✗ **Per-slice DER** — 每個 slice 樣本太少（10-15），統計信心弱
5. ✗ **Role attribution 結論** — v1 完全沒測

### 預期 v2 跑完後 DER 變化

| Model | v1 DER (auto) | v2 預期 DER (fixed2 + new GT) | 變化原因 |
|---|---|---|---|
| pyannote 3.1 | 29.46% | **12-18%** | num_speakers + GT 雙修 |
| pyannote community-1 | 28.57% | **10-15%** | 同上 + community-1 更強 |
| Sortformer v1 | 27.52% | **18-25%** | GT 改一點，但 Sortformer 對 Mandarin/Hokkien 弱 |
| CAM++ | 12.13% | **6-10%** | 本來就保守，GT 修了會降 |
| ERes2NetV2 patched | 12.13%（同 CAM++，patch 沒真生效）| **5-9%** | 真正 patch + 假設 ERes2NetV2 比 CAM++ 強 1-3pp |

---

## v1 工作可重用的部分

| v1 資產 | v2 重用方式 |
|---|---|
| 90 條 dialog scripts | 部分情境取出當 v2 dialog spec 的 inspiration |
| `phase5_dialog_scripts.py` 文案 | 重新拆出來，加 schema metadata |
| TTS 合成腳本 | 重寫成 `generate_tts_and_gt.py`，加 VAD-trim |
| `noise.py` | 直接複用 |
| `phase5_06_eres2netv2_v2.ipynb` 的 patch 邏輯 | 抽出來放進 `run_diarization.py`，加 sanity check |
| 已下載的 model weights cache | 複用，省下重新下載時間 |
| 4 model 既有 results CSV | **不寫進 v2 final report**，但保留作 reference / before-after 對比 |

---

## 工時 / 成本對比

| 維度 | v1（已完成）| v2（計劃）|
|---|---|---|
| 工時 | ~1 週 | **3-4 週** |
| TTS API calls | ~700 | ~2,500 |
| Dialog 數 | 90 | 270 |
| Model inference 數 | ~360 | ~1,890 |
| 最終 report 可不可信 | △ 內部 PoC ✓，外部報告 ✗ | ✓ 可寫進正式 production-readiness 文件 |

---

## 建議

1. **如果只需要內部「pyannote 該不該換」的快速決策** → v1 結論「CAM++ 12% 顯著優於 pyannote 29%」已經夠
2. **如果要寫進 KGI 正式 production-readiness 報告** → 必須走 v2
3. **如果只想 fix 一兩個最關鍵 bug**（不做整套）→
   - 最小修正：跑 pyannote `num_speakers=2` + 修 GT VAD-trim
   - 預期 v1 4-model DER 數字會大幅改善（pyannote 29% → 15%）
   - 但 dataset 規模 + slice 涵蓋面不夠（仍只 90 條），不適合外部報告

---

_Doc version: 1.0 / 2026-05-22_
