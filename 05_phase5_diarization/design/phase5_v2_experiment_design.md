# Phase 5 v2 — KGI 客服語者分離完整實驗設計

**版本**：v2.0
**前身**：`benchmark/phase5_diarization`（v1，已部分執行）
**核心改進**：採納 codex AI 點出的方法論問題 + 保留 v1 已驗證的工程實作

---

## 0. 摘要：為什麼有 v2

v1 在 90 條 TTS 對話上跑出了 4-model 對比（CAM++ 12.13%、Sortformer 27.52%、pyannote 28-29%）。但獨立 AI 審查（codex）指出 **3 個結論可能不可信的方法論 bug**：

| Bug | 影響 |
|---|---|
| **GT 把 TTS turn 完整時長當 speech activity**，但 TTS 音檔頭尾跟句中有大量 silence（量化分析顯示 GT 區間僅 65.8% frame 真的在發聲）| 模型正確偵測 silence 反被算成 missed speech → 嚴格抓 speech activity 的 pyannote / Sortformer **DER 被高估 10-15pp**。CAM++ 12% 可能其實是 6-8% |
| **pyannote 用 automatic speaker count**，沒設 `num_speakers=2` | 22% case 預測錯人數（1 or 3） → DER 直接被拖垮 |
| **沒做 dev/test split** | 整套 90 條都當 evaluation，等於沒留 tune 空間，**結果不能直接用來下「pyannote vs SOTA 誰好」結論** |

v2 重新設計目標：
1. **修 GT 方法論**：分 `speech` 跟 `turn` 兩套 RTTM
2. **修 pyannote 設定**：強制 `num_speakers=2`，公平 baseline
3. **加 dev/test split**：tune 跟 eval 分離
4. **擴充 slice**：補 same_gender、customer_barge_in、simultaneous_start、third_party 等
5. **加 stress conditions**：codec / echo / babble 變體
6. **公平 role attribution 子實驗**：4 種 baseline 分別評估

**設計取捨**：在 codex 的學術嚴謹（520 dialogs）跟我 v1 的快速 PoC（90 dialogs）之間取**~270 dialogs 中間值**，平衡 TTS 成本、Colab session 時間、結論可信度。

---

## 1. 研究問題（重申 + 精確化）

### RQ1：跨段同人識別（re-entry consistency）

模型能否在以下情境保持同一 speaker label：

- 短停頓（2-10s）後客戶回來
- 長停頓（30-120s）後客戶回來
- 跨多個 turn 後客戶回來
- 短應答（「好」「嗯」「請稍等」）是否被誤分或拆出新 speaker

**Pass 標準**：跨段一致性 `same_speaker_link_accuracy > 0.90`，`speaker_fragmentation_rate < 0.15`

### RQ2：重疊語音處理（overlap detection）

分三級評估：

| 等級 | 任務 | 業務用途 |
|---|---|---|
| **L1** Overlap 偵測 | 「這段有重疊」 | 插話率、人工複核提示 |
| **L2** 重疊段識別兩個 speaker | 「此段 A 跟 B 都 active」 | 互動分析、發言時長 |
| **L3** Word-level 歸屬 | 「重疊段 A 講了 X、B 講了 Y」 | 高品質逐字稿、合規 |

L3 需要 ASR + word timestamp alignment，**Phase 5 v2 只測 L1/L2**。L3 列為 Phase 6 工作。

**Pass 標準**：`overlap_detection_recall > 0.70`、`two_speaker_active_ratio_in_overlap > 0.60`

### RQ3：Agent vs Customer 角色辨識

**不是**辨識「哪位客服」（個人識別），**是**辨識「這個匿名 speaker cluster 是 agent 還是 customer」（角色分類）。

四種 baseline 各自評估：

1. **Conversation-rule**：先講話且說「您好凱基」「敝姓」「請問身分證」的 cluster = agent
2. **ASR text classifier**：把每個 cluster 前 N 句 ASR 文字丟 GPT-4o-mini 判斷
3. **Acoustic agent-class prior**：用 enrolled agent voices 算 cosine similarity
4. **Fusion**：規則 + 文字 + 聲學分數加權

**Pass 標準**：`role_accuracy_time_weighted > 0.95`（任一 baseline），fusion 期望 > 0.98。

---

## 2. 資料集架構

### 2.1 三層 split — **嚴格隔離**

| Split | 規模 | 用途 | 規則 |
|---|---|---|---|
| `dev_smoke` | **20** dialogs | Pipeline 驗證、debug | 早期跑 1-2 個模型確認 GT/inference 流程通 |
| `dev_tune` | **30** dialogs | Threshold/post-processing 調參 | 可重複跑、可調參，但**不准進 final 報告** |
| `test_core` | **120** dialogs | 最終 clean DER 評估 | **凍結**，模型開跑後不能改 GT/audio |
| `test_stress` | **60** audio variants | Noise robustness（重用 test_core dialogs 加 noise）| 同 audio_dur，GT 不變 |
| `role_attribution` | **40** dialogs（部分與 test_core 重疊）| 角色分類 4-baseline 評估 | 額外標 enrollment refs |

**總計：270 unique audio files（含 60 stress 變體）**

合理性論證：
- v1 90 條已經夠看 model ranking，但每個 slice 只 10-15 條 → 統計信心弱
- codex 提議 520 條過大（TTS API + 評估時間爆炸）
- 270 條每個 slice 約 10-16 條 + dev/test 嚴格分離 + stress 變體 = 平衡點

### 2.2 Test_core 12 個 Slice（120 dialogs）

| Slice | N | 主測什麼 | RQ |
|---|---|---|---|
| `clean_turn_taking` | 16 | 基本 baseline | 1, 3 |
| `reentry_short_gap` | 12 | 短停頓後同人 | 1 |
| `reentry_long_gap` | 12 | 長停頓（30-120s）後同人 | 1 |
| `agent_barge_in` | 12 | 客服打斷客戶 | 2 |
| `customer_barge_in` | 12 | 客戶打斷客服（v1 沒有的）| 2 |
| `simultaneous_start` | 10 | 兩人同時開口（差 < 300ms）| 2 |
| `short_backchannel` | 10 | 「嗯」「好」短回應 | 1 |
| `hokkien_pure` | 10 | 純台語 | 1, 3 |
| `codeswitch` | 10 | 國語+台語混雜 | 1, 3 |
| `same_gender_similar` | 10 | 兩人同性別、聲音接近 | 1 |
| `third_party_background` | 6 | 家人在旁邊插話（不該歸客服或客戶）| 3 |

### 2.3 Stress conditions（60 variants）

從 test_core 隨機抽 dialog，套 noise transform：

| 條件 | N | 取樣源 |
|---|---|---|
| `codec_g711`（電話壓縮）| 20 | 隨機抽 20 條 test_core |
| `echo_500ms`（迴聲）| 20 | 隨機抽 20 條 |
| `babble_15db_snr`（人聲背景）| 10 | 隨機抽 10 條 |
| `low_volume_imbalance`（單邊降 6dB 模擬距離差）| 10 | 從 overlap 子集挑 |

複用 `benchmark/noise.py`（v1 已有的工具）。**GT timing 不變**（noise 不改音檔長度）。

### 2.4 Role_attribution 集（40 dialogs）

- 30 條重用 test_core 的 clean / overlap / hokkien / same_gender 等代表 case
- 10 條新增「impostor / unknown」場景：
  - 5 條：第三方加入，role=`other`（測 unknown rejection）
  - 5 條：客戶聲音剛好接近 enrolled agent pool 中某員工（測 false positive）

額外標 `role_attribution_manifest.csv`，含每個 cluster 的 gold role + reference enrollment pool 路徑。

---

## 3. Dialog Spec 規格（JSON Schema）

### 3.1 完整 spec 範例

```json
{
  "case_id": "v2_smk001",
  "split": "dev_smoke",
  "slice": "clean_turn_taking",
  "language_profile": "mandarin",
  "scenario": "住院理賠文件確認",
  "recording_start": "full_call",
  "role_signal_strength": "strong",
  "duration_target_sec": 35,
  "participants": {
    "agent": {
      "role": "agent",
      "gender": "F",
      "persona": "calm_claims_agent",
      "voice_pool": "azure_zh_tw_female",
      "voice_id": "zh-TW-HsiaoChenNeural"
    },
    "customer": {
      "role": "customer",
      "gender": "M",
      "persona": "middle_aged_policyholder",
      "voice_pool": "azure_zh_tw_male",
      "voice_id": "zh-TW-YunJheNeural"
    }
  },
  "turns": [
    {
      "turn_idx": 0,
      "speaker": "agent",
      "role": "agent",
      "language": "mandarin",
      "text": "您好凱基人壽客服中心，敝姓林，很高興為您服務。",
      "tags": ["greeting", "service_script"],
      "timing": {"delay_before_sec": 0.0}
    },
    {
      "turn_idx": 1,
      "speaker": "customer",
      "role": "customer",
      "language": "mandarin",
      "text": "您好，我想確認住院理賠要準備哪些資料。",
      "tags": ["request"],
      "timing": {"delay_before_sec": 0.45}
    }
    // ...
  ],
  "metadata": {
    "schema_version": "2.0",
    "created_at": "2026-05-22",
    "audio_format": "linear16/16khz/mono"
  }
}
```

### 3.2 Schema 重點

| 欄位 | 用途 |
|---|---|
| `role_signal_strength` | `strong` / `medium` / `weak` — role attribution 評估會按這分層 |
| `recording_start` | `full_call` / `mid_call` — mid_call 沒有 greeting，rule-based role attribution 失效 |
| `persona` | 內部分析用，不影響 TTS（只是 voice 選擇 hint） |
| `voice_id` | 明確指定 TTS voice（v1 是 voice_pool 自動選，v2 dialog spec 直接指定確保可重現）|
| `tags` per turn | `greeting`、`pii_self_report`、`agent_barge_in`、`reentry`、`backchannel`、`third_party_background` 等 |
| `timing.delay_before_sec` | 與前一 turn 結束的延遲 |
| `timing.overlap_with_prev` + `overlap_offset_sec` | 重疊區段（offset 負值 = 提早開始）|
| `timing.forced_pause_after_sec` | 長停頓（reentry slice 用）|

### 3.3 Voice Pool 規格（v1 經驗）

| Pool | Voices | 用途 |
|---|---|---|
| `azure_zh_tw_female` | HsiaoChenNeural, HsiaoYuNeural | Mandarin 女客服/女客戶 |
| `azure_zh_tw_male` | YunJheNeural | Mandarin 男（唯一）|
| `azure_zh_tw_female_alt2` | HsiaoYuNeural（重複利用做 same_gender 對比）| 同性別 case |
| `yating_tai_female` | tai_female_1 | Hokkien 女 |
| `yating_tai_male` | tai_male_1 | Hokkien 男 |
| `azure_zh_tw_third_party` | HsiaoChenNeural（不同 prosody）| 第三方 |

**Same-gender stress slice 規則**：兩個 voice **同性別但不同 voice id**（例如 HsiaoChen + HsiaoYu），不要用「同一 voice 加 prosody」假裝兩個人。

---

## 4. TTS + Ground Truth Pipeline（核心方法論改進）

### 4.1 Flow（v1 vs v2）

```
v1 (有 bug):
  TTS each turn → 用 full wav duration 當 speaker activity → mix → RTTM 從 full turn 寫
  
v2 (修正):
  TTS each turn → 存 raw → VAD 偵測實際發聲段 → trim 頭尾 silence → 記錄句中 pause
  → schedule 排上 master timeline → mix mono → 寫三種 GT
```

### 4.2 VAD 參數（具體數字，要 commit 到 spec）

```yaml
vad:
  frame_length_ms: 20
  hop_ms: 10
  rms_absolute_floor: 120        # int16 scale, 經驗值
  rms_relative_factor: 0.08      # noise_p10 + factor * (p95 - p10)
  merge_gap_max_sec: 0.150       # 短於此的 gap 合併（保留自然停頓 < 150ms）
  drop_island_min_sec: 0.080     # 短於此的活躍段丟掉（雜音）
  drop_island_backchannel_sec: 0.050  # backchannel-tagged turn 用更寬鬆
  pad_each_side_sec: 0.060       # 偵測到的邊界往外 pad（呼吸聲）
```

**重要**：所有 dev_smoke audio 跑完 VAD 後**人工抽 12 條 spot check**（畫 waveform + RTTM 對照），確認 GT 跟實際發聲段對齊，才開始 test_core 全量生成。

### 4.3 三種 GT artifact

| 檔案 | 內容 | 用於 |
|---|---|---|
| `ground_truth_speech.rttm` | 每個 turn VAD 偵測到的實際 speech-active intervals | **標準 DER**（DER_strict、DER_lenient_025）|
| `ground_truth_turn.rttm` | 每個 turn 的 trimmed start/end（去頭尾 silence 但保留句中 pause）| Turn ownership、role attribution |
| `overlap_regions.json` | 精確 overlap intervals + 兩 speaker | DER_overlap_only metric |

### 4.4 UEM 檔案（評估用 mask）

| UEM | 範圍 | 用於 |
|---|---|---|
| `uem_full.uem` | 整段音檔（0 - duration）| 標準 DER |
| `uem_no_overlap.uem` | 排除 overlap regions | DER_no_overlap |
| `uem_overlap_only.uem` | 只含 overlap regions | DER_overlap_only |

### 4.5 額外 manifest

```csv
case_id,split,slice,language,duration_sec,n_turns,n_overlaps,
vad_active_ratio,trim_total_sec,
audio_path,gt_speech_rttm,gt_turn_rttm,
overlap_json,uem_full,uem_no_overlap,uem_overlap_only,
spec_path
```

`vad_active_ratio`：speech-active 時長 / total duration（應該 0.6-0.85 區間，太高 = 沒 silence，太低 = TTS 異常）

---

## 5. 模型對比矩陣

### 5.1 必跑 local models（7 種 config）

| ID | Family | Model | Config | License | 預期 DER |
|---|---|---|---|---|---|
| **M1a** | pyannote | speaker-diarization-3.1 | `num_speakers=2` | CC-BY-4.0 | 8-12% |
| **M1b** | pyannote | speaker-diarization-3.1 | auto speaker count | CC-BY-4.0 | 12-18% |
| **M2a** | pyannote | speaker-diarization-community-1 | `num_speakers=2` + exclusive | CC-BY-4.0 | 7-11% |
| **M2b** | pyannote | speaker-diarization-community-1 | auto | CC-BY-4.0 | 11-16% |
| **M3** | 3D-Speaker | CAM++ pipeline (`speech_campplus_speaker-diarization_common`) | default | Apache 2.0 | 6-10%（v1 已驗證 12.13% with broken GT）|
| **M4** | 3D-Speaker | ERes2NetV2 (patched into CAM++ pipeline) | patch infer_diarization.py | Apache 2.0 | 5-9%（理論上 < CAM++） |
| **M5** | NVIDIA | diar_sortformer_4spk-v1 offline | default | CC-BY-4.0 | 10-20% (Hokkien 可能崩) |

### 5.2 Stretch（時間允許才跑）

| ID | Model | 為何 stretch |
|---|---|---|
| M6 | NVIDIA Streaming Sortformer v2.1 offline mode | 看 v2 比 v1 好多少 |
| M7 | DiariZen WavLM-large (`BUT-FIT/diarizen-wavlm-large-s80-md`) | **CC BY-NC 4.0 不可商用**，僅作 research upper bound |
| M8 | pyannoteAI Precision-2 (商用 API) | 上限參考 |

### 5.3 已知 library issues（不重複踩雷）

| Model | Issue | 結論 |
|---|---|---|
| FunASR SOND | modelscope 跟 funasr 版本相容性壞掉 | **不測**，文件記錄為 "library issue, deferred" |
| `ms_pipeline(sv_model=...)` for CAM++ | 對 CAM++ 是 silent no-op，**會 silently 落到 CAM++ default embedder** | **不可信**，要走 patch source code 路線 |
| Diarization3Dspeaker CAM++ hardcode | embedder 寫死，要 monkey-patch | v2 用 `Diarization3DspeakerERes` subclass（v1 notebook 06 的方法）|

### 5.4 Config 變體

每個 pyannote model 跑兩個 config（auto + fixed2），公平驗證「pyannote 不行」是 `num_speakers` 沒設好還是 model 本身的限制。

---

## 6. 評估指標完整列表

### 6.1 Diarization metrics（必算）

| Metric | Collar | UEM | 用途 |
|---|---|---|---|
| `DER_strict` | 0.0s | full | 嚴格學術 benchmark |
| `DER_lenient_025` | 0.25s | full | 業界慣例 |
| `DER_no_overlap` | 0.25s | no_overlap | 排除 overlap 後的「乾淨段」DER |
| `DER_overlap_only` | 0.25s | overlap_only | **重疊段獨立分數** |
| `JER` | - | full | Jaccard Error — speaker coverage balance |
| `missed_speech_rate` | - | full | DER 三分量之一 |
| `false_alarm_rate` | - | full | DER 三分量之一 |
| `speaker_confusion_rate` | - | full | **業務最關鍵分量**（誰歸錯人） |
| `speaker_count_accuracy` | - | - | 正確猜 2 個語者的比例 |

### 6.2 Consistency metrics（RQ1）

| Metric | 計算方式 |
|---|---|
| `same_speaker_link_accuracy` | 同一 GT speaker 的所有 turn 是否被分到同一 cluster |
| `speaker_fragmentation_rate` | 一個 GT speaker 被分裂成 N 個 hyp cluster 的平均 N |
| `cluster_purity` | hyp cluster 中最多數的 GT speaker 比例 |
| `label_swap_count` | 跨段 speaker label 切換次數（與 GT 對齊後）|

### 6.3 Overlap metrics（RQ2）

| Metric | 計算方式 |
|---|---|
| `overlap_detection_recall` | GT overlap 區段中，hyp 至少偵測到 2 個 active speaker 的比例 |
| `overlap_detection_precision` | hyp 偵測到 overlap 的區段中，GT 也是 overlap 的比例 |
| `two_speaker_active_ratio_in_overlap` | overlap 區段內，平均同時 active 的 speaker 數 |
| `overlap_F1` | recall+precision 的調和平均 |

### 6.4 Role attribution metrics（RQ3）

| Metric | 用途 |
|---|---|
| `role_accuracy_time_weighted` | 按 speech duration 加權 |
| `role_accuracy_turn_weighted` | 按 turn 數加權（避免長 turn 主導）|
| `agent_role_precision` | hyp 標 agent 中真的是 agent 比例 |
| `agent_role_recall` | GT agent 被標 agent 比例 |
| `customer_as_agent_error` | 客戶被誤標客服比例（**業務 critical**：可能洩漏錯誤的合規追責方向）|
| `agent_as_customer_error` | 客服被誤標客戶 |
| `third_party_as_agent_error` | 第三方被誤標客服 |
| `unknown_rejection_rate` | role_attribution split 中 impostor cluster 被識別為 unknown 的比例 |

### 6.5 Operational metrics

| Metric | 用途 |
|---|---|
| `inference_time_sec` | 單條音檔推論時間 |
| `RTF` | inference_time / audio_duration |
| `peak_gpu_memory_mb` | GPU 占用 |
| `model_size_mb` | 磁碟 footprint |

---

## 7. Role Attribution 子實驗（RQ3 詳細設計）

### 7.1 四個 baseline

#### Baseline A：Conversation-Rule

```python
def rule_based_role(cluster_turns, dialog_metadata):
    """
    First-speaker heuristic + greeting keyword.
    
    - 如果第一個 turn 是這個 cluster → 給 agent prior
    - 如果 cluster 任一 turn 含 [「您好凱基」「敝姓」「為您服務」「請問身分證」]
      → strong agent prior
    - 否則 customer prior
    """
```

對 `role_signal_strength=strong` case 應該接近完美。`mid_call` recording 沒 greeting 會崩。

#### Baseline B：ASR Text Classifier

```python
def text_role(cluster_text, gpt_client):
    """
    把 cluster 前 5 句 ASR 文字丟 GPT-4o-mini：
    'Given these utterances, is the speaker an insurance customer service agent or a customer?'
    """
```

Phase 3 既有 Azure GPT-4o-mini 直接用，零新依賴。

#### Baseline C：Acoustic Agent-Class Prior

```python
def acoustic_role(cluster_embedding, agent_pool_embeddings, threshold=0.7):
    """
    cosine similarity between cluster embedding and pool of agent enrollment embeddings.
    
    > threshold → agent
    < threshold → customer  
    very_low + matches none → other / unknown
    """
```

需要 enrollment pool。v2 用 8-12 個 Azure voice 模擬「不同客服」（同 voice family 但不同 prosody），給 role_attribution split 用。

#### Baseline D：Fusion

```python
def fusion_role(cluster, rule_score, text_score, acoustic_score, weights=(0.3, 0.3, 0.4)):
    """
    Weighted softmax of three baselines.
    """
```

### 7.2 評估流程

1. 跑 best diarization model（預期 ERes2NetV2 patched）→ 取 cluster
2. 對每個 cluster 跑 4 個 baseline
3. 計算 §6.4 所有 role metrics
4. 報表：哪個 baseline 在 `role_signal_strength=weak` / `mid_call` / `same_gender` 條件下表現最穩

---

## 8. 執行階段（Stages 與 Gates）

### Stage A：dev_smoke 驗證（20 dialogs）

**Goal**：確認 pipeline 通、GT 合理

**Steps**：
1. 寫 20 條 dev_smoke spec（涵蓋所有 12 slice）
2. 跑 TTS + VAD-based GT 生成
3. 人工 spot-check 12 條 audio + waveform
4. 跑 M3 (CAM++) 跟 M1a (pyannote num_speakers=2) 各一輪
5. 對照 GT 速看 DER

**Gate**：
- VAD active ratio 在 0.6-0.85 區間
- pyannote DER < 15%（修了 num_speakers + GT 後應該大幅改善）
- 沒有 systematic GT 錯誤

**失敗動作**：回頭調 VAD 參數，不准進 Stage B

### Stage B：test_core 全量生成（120 dialogs）

**Goal**：產生最終評估集

**Steps**：
1. 寫剩下 100 條 spec
2. 批次 TTS + VAD + 三 GT
3. 凍結 audio + GT（重新跑用 git tag 版本）

**Gate**：
- 12 slice 規模都符合計劃
- 每個 slice 至少 1 條人工抽檢

### Stage C：必跑 model matrix（7 個 config）

**Goal**：得到正式 DER 比較

**Steps**：
1. 跑 M1a/M1b/M2a/M2b/M3/M4/M5 共 7 個 config
2. 全部存 detections + raw_outputs + model_metadata
3. 計算 §6 所有 metrics
4. 寫 cross-model summary CSV

**Gate**：
- 7 個 config 都跑完 120 條
- 統計 paired t-test 確認 model 間差距是否 significant

### Stage D：Stress 變體（60 audio）

**Goal**：抗噪健壯性

**Steps**：
1. 從 test_core 隨機抽生成 stress variants
2. 跑 best 3 model（從 Stage C 結果挑）on 60 stress audio
3. 計算 DER delta（stress vs clean 同樣 dialog）

### Stage E：Role Attribution 子實驗（40 dialogs）

**Goal**：4 baseline 完整對比

**Steps**：
1. Build agent enrollment pool（8-12 個 voice 樣本）
2. 用 Stage C best model output 取 cluster
3. 跑 4 baseline，算 §6.4 所有 metrics
4. 報告 fusion 是否顯著 > 單一 baseline

### Stage F：報告撰寫 + Phase 5 v2 final report

- Cross-model DER table
- Per-slice breakdown
- Role attribution 4-baseline 對比
- Production readiness gate（DER < 12% 為 acceptance）

### Stretch Stage G（時間允許）

- M6/M7/M8（Sortformer v2 / DiariZen / pyannoteAI）
- 真實電話樣本 sanity check（如能取得 5-10 條去識別化錄音）

---

## 9. 檔案結構

```
benchmark/phase5_v2/
├── design/
│   ├── phase5_v2_experiment_design.md         ← 本檔案
│   └── dialog_spec.schema.json                 ← JSON Schema 驗證
├── data/
│   ├── dialog_specs/
│   │   ├── dev_smoke.json                      ← 20 specs
│   │   ├── dev_tune.json                       ← 30 specs
│   │   ├── test_core.json                      ← 120 specs
│   │   └── role_attribution.json               ← 40 specs
│   ├── manifest.csv                            ← 整合 manifest
│   ├── ground_truth_speech.rttm                ← 全部 case 的 speech RTTM
│   ├── ground_truth_turn.rttm                  ← 全部 case 的 turn RTTM
│   ├── ground_truth_per_case/
│   │   ├── speech/{case_id}.rttm
│   │   └── turn/{case_id}.rttm
│   ├── overlap_regions.json
│   ├── uem/
│   │   ├── full/{case_id}.uem
│   │   ├── no_overlap/{case_id}.uem
│   │   └── overlap_only/{case_id}.uem
│   └── role_attribution_manifest.csv
├── audio/
│   ├── dialogs_clean/{case_id}.wav             ← 220 clean wavs
│   ├── dialogs_stress/{case_id}_{condition}.wav  ← 60 stress variants
│   ├── turns_raw/{case_id}/turn_XX_{speaker}.wav  ← per-turn raw（VAD 用）
│   ├── turns_trimmed/{case_id}/turn_XX.wav     ← trimmed 後（mixer 用）
│   ├── enrollment/                              ← 8-12 個 agent reference voices
│   │   ├── agent_001.wav … agent_012.wav
│   │   └── enrollment_manifest.csv
│   └── README.md
├── scripts/
│   ├── generate_dialog_specs.py                ← 從 template + scenarios 生 spec JSON
│   ├── generate_tts_and_gt.py                  ← TTS + VAD + GT pipeline
│   ├── apply_noise_transforms.py               ← Stress 變體生成
│   ├── build_enrollment_pool.py                ← 抽 agent voice + 算 embedding
│   ├── run_diarization.py                      ← 統一 model runner（支援 7 個 config）
│   ├── eval_diarization.py                     ← §6.1-6.3 metrics
│   ├── eval_role_attribution.py                ← §6.4 4-baseline
│   ├── role_classifiers/
│   │   ├── rule_based.py
│   │   ├── text_classifier.py
│   │   ├── acoustic_prior.py
│   │   └── fusion.py
│   └── notebooks/
│       ├── stage_a_dev_smoke.ipynb
│       ├── stage_b_test_core.ipynb
│       ├── stage_c_model_matrix.ipynb
│       ├── stage_d_stress.ipynb
│       ├── stage_e_role_attribution.ipynb
│       └── stage_f_final_summary.ipynb
├── results/
│   ├── detections/{model_id}/{case_id}.rttm
│   ├── detections/{model_id}/{case_id}.json
│   ├── runtime/{model_id}.jsonl
│   ├── model_metadata/{model_id}.json
│   ├── diarization_metrics.csv                  ← per-case × per-model
│   ├── consistency_metrics.csv
│   ├── overlap_metrics.csv
│   ├── role_attribution_metrics.csv
│   └── cross_model_summary.csv
└── reports/
    ├── phase5_v2_final_report.md
    ├── stage_a_smoke_report.md
    ├── stage_c_model_comparison.md
    └── role_attribution_analysis.md
```

---

## 10. 可重現性要求

### 10.1 每個 model run 必存

```json
// model_metadata/{model_id}.json
{
  "model_id": "pyannote_31_fixed2",
  "model_repo": "pyannote/speaker-diarization-3.1",
  "model_revision": "main",
  "model_sha256": "...",
  "config": {
    "num_speakers": 2,
    "min_speakers": 2,
    "max_speakers": 2
  },
  "dependencies": {
    "pyannote.audio": "3.4.0",
    "torch": "2.10.0+cu128",
    "modelscope": "1.32.x"
  },
  "hardware": {
    "gpu": "Tesla T4",
    "vram_used_mb": 4234
  },
  "runtime": {
    "load_sec": 7.4,
    "total_inference_sec": 543.2,
    "mean_rtf": 0.045
  },
  "license_note": "CC-BY-4.0",
  "tuned_on_split": null,
  "command": "python run_diarization.py --model pyannote_31_fixed2 --split test_core"
}
```

### 10.2 Git 控管

- 所有 spec + script 進 git
- audio 大檔不進（在 README 標明 Drive 路徑）
- 模型 weights cache 在 Drive 共享路徑（不重複下載）

### 10.3 Notebook 規範

- 不准只用 in-memory state（重啟 runtime 也要能重現）
- 每個 stage 一個 notebook
- 結果寫死到檔案，下一個 stage notebook 讀檔案

---

## 11. 預估時程與成本

### 11.1 工作分解

| Stage | 工作 | 工時 | API/GPU 成本 |
|---|---|---|---|
| Design + Spec | 寫 dev_smoke (20) + 設計 schema | **1 天** | 0 |
| Stage A | TTS 20 + GT + spot check + run 2 model | **2 天** | NT$5 |
| Spec | 寫 dev_tune (30) + test_core (120) + role_attr (40) | **2 天** | 0 |
| Stage B | 批次 TTS 190 dialogs + 60 stress | **2 天** | NT$30 |
| Stage C | 7 model × 120 = 840 inference | **3 天**（多個 Colab session）| NT$0（GPU 免費 Colab）|
| Stage D | 3 model × 60 stress | **1 天** | NT$0 |
| Stage E | Role attribution 4-baseline | **2 天** | NT$50（GPT-4o-mini text classifier）|
| Stage F | 報告整合 | **2 天** | 0 |
| Stretch (G) | M6-M8 + 真實電話 | **3-5 天** | NT$200（pyannoteAI 商用 API trial）|

**主路徑 ~15 工作天 = 3-4 週**
**含 stretch ~20 工作天 = 4-5 週**

### 11.2 與 v1 比較

| 維度 | v1 | v2 |
|---|---|---|
| Dialog 數 | 90 | 270（120 core + 60 stress + 30 tune + 40 role + 20 smoke）|
| Model configs | 4 | 7 必跑 + 3 stretch |
| Metrics | ~5 | 21 |
| Splits | 0 | 4 |
| Stress variants | 0 | 60 |
| 工時 | 1 週執行 | **3-4 週** |

---

## 12. Decision Criteria（v2 acceptance gates）

### 12.1 Diarization quality

| Metric | Pass | Stretch |
|---|---|---|
| Best model DER_lenient_025 (test_core) | **< 12%** | < 8% |
| Best model DER_no_overlap | < 10% | < 6% |
| Best model speaker_count_accuracy | > 95% | > 99% |
| pyannote num_speakers=2 vs auto | auto 應該 < +5pp | - |

### 12.2 Consistency (RQ1)

| Metric | Pass |
|---|---|
| same_speaker_link_accuracy | > 0.90 |
| speaker_fragmentation_rate | < 0.20 |

### 12.3 Overlap (RQ2)

| Metric | Pass |
|---|---|
| overlap_detection_recall | > 0.70 |
| DER_overlap_only | < 25% |

### 12.4 Role attribution (RQ3)

| Metric | Pass |
|---|---|
| Fusion role_accuracy_time_weighted | > 0.95 |
| Best single baseline | > 0.85 |
| customer_as_agent_error | < 5% |

### 12.5 Stress robustness

| Metric | Pass |
|---|---|
| Best model DER delta (codec - clean) | < +5pp |
| Best model DER delta (echo - clean) | < +10pp |

---

## 13. 風險與緩解

| 風險 | 機率 | 影響 | 緩解 |
|---|---|---|---|
| VAD 參數調太緊 → 切掉真實 speech | 中 | 高 | dev_smoke spot check 12 條，看不對立刻調 |
| TTS API rate limit（270 dialogs）| 中 | 中 | 分散時段、用兩個 API key、retry with backoff |
| Colab session 12hr timeout | 高 | 中 | 每個 stage 一個 session，跑前用 checkpoint 機制 |
| ERes2NetV2 patch 不穩 | 中 | 高 | 已在 v1 notebook 06 驗證 + sanity check Delta check |
| Role attribution acoustic prior 對 Hokkien 失效 | 中 | 中 | Acoustic baseline 只看 Mandarin 部分 |
| 真實電話 vs TTS domain gap | 高 | 高 | 報告中明標 limit；stretch stage 收 5 條真實 sanity |
| Library version 衝突 | 低 | 低 | 已在 v1 解決，沿用穩定組合 |

---

## 14. v1 結論的「可信度轉換」

v1 跑出的 4-model DER 經 v2 GT 重新評估後，預期變化：

| v1 model | v1 DER | v2 預期 DER（修 GT + num_speakers=2）| 解讀 |
|---|---|---|---|
| pyannote 3.1 (auto) | 29.46% | **預期 12-18%** | num_speakers + GT 雙修，大降 |
| pyannote community-1 (auto) | 28.57% | **預期 10-15%** | 同上 |
| Sortformer v1 | 27.52% | **預期 18-25%**（Sortformer 對 Mandarin 弱跟 GT 無關，仍會偏高）| GT 改善有限 |
| CAM++ | 12.13% | **預期 6-10%** | 本來就跑得保守，GT 修了會降 |
| ERes2NetV2 (patched) | 跟 CAM++ 一樣（patch 沒驗證真生效）| **需重跑** | 必須在 v2 環境重做 |

**結論**：v1 的 model ranking 大致對（CAM++ > Sortformer > pyannote），但 absolute DER 數字**不能拿去寫 production 報告**。

---

## 15. 跟 v1 的關係

| 項目 | v1 處理 | v2 處理 |
|---|---|---|
| v1 90 條 audio + GT | 保留作對照組 | 不重用（GT 方法論不同）|
| v1 dialog scripts | 部分情境可重用作 inspiration | 整體 spec 重寫（加 schema）|
| v1 noise.py | 直接複用 | 沿用，加入 stress flow |
| v1 generate_phase5_dialog_tts.py | 作 reference implementation | 重寫，整合 VAD-trim |
| v1 build_ground_truth.py | **棄用**（有 bug） | 重寫，輸出三種 GT |
| v1 phase5_06_eres2netv2_v2.ipynb | 保留 patch 邏輯 | 整合進 run_diarization.py |
| v1 model results CSV | 保留為 "v1 reference"，**不寫進最終報告** | v2 自己重跑 |

---

## 16. 立即可做的 Stage 0（本週可開始）

| Day | 工作 | 產出 |
|---|---|---|
| 1 | 寫 `dialog_spec.schema.json` + 20 條 dev_smoke spec | data/dialog_specs/dev_smoke.json |
| 2 | 寫 `generate_tts_and_gt.py`（含 VAD-trim） | scripts/generate_tts_and_gt.py |
| 3 | 跑 dev_smoke + spot check 12 條 | audio/dialogs_clean/*.wav, data/ground_truth_*.rttm |
| 4 | 跑 M3 (CAM++) + M1a (pyannote num_speakers=2) on dev_smoke | results/.../dev_smoke 對照 |
| 5 | Stage A gate review，決定是否 proceed Stage B | reports/stage_a_smoke_report.md |

---

## 17. 主要參考文獻

- [pyannote-audio diarization-3.1 / community-1 model cards](https://huggingface.co/pyannote)
- [3D-Speaker repo (ERes2NetV2 + diarization recipe)](https://github.com/modelscope/3D-Speaker)
- [NVIDIA Sortformer model cards](https://huggingface.co/nvidia/diar_sortformer_4spk-v1)
- [DiariZen WavLM model card](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md) — CC BY-NC 4.0
- [SOND ModelScope](https://modelscope.cn/models/iic/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch) — library compat issue
- pyannote.metrics: DER / JER 計算
- Codex AI Phase 5 review report（v2 多項設計取自此份）
- v1 Phase 5 phase5_diarization 既有實作（v2 的 baseline reference）

---

_本文件版本：v2.0 / 2026-05-22_
_作者：Claude_
_狀態：Design approved, awaiting implementation_
