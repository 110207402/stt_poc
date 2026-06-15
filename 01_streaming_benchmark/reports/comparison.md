# STT Model Comparison Report

Generated: 2026-03-22 10:02:01

Total runs: 1800 | OK: 1800 | Failed: 0
Noise conditions tested: clean, pink(20dB), white(20dB)

## Overall Ranking (sorted by CER → E2E)

| Rank | Model | Noise | CER (%) | TTFF (ms) | E2E (ms) | RTF | P95 Lat (ms) | Success |
|------|-------|-------|---------|-----------|----------|-----|-------------|---------|
| 1 | zipformer-zh-xl | clean | 10.70 | 8615 | 42023 | 1.038 |  | 100.0% |
| 2 | zipformer-zh-xl-t | clean | 10.93 | 8559 | 42100 | 1.040 |  | 100.0% |
| 3 | zipformer-zh-xl | pink(20dB) | 11.11 | 8637 | 42056 | 1.039 |  | 100.0% |
| 4 | paraformer | pink(20dB) | 11.13 | 8517 | 41873 | 1.034 |  | 100.0% |
| 5 | paraformer | clean | 11.17 | 8500 | 41841 | 1.034 |  | 100.0% |
| 6 | paraformer | white(20dB) | 11.32 | 8495 | 41897 | 1.035 |  | 100.0% |
| 7 | zipformer-zh-xl | white(20dB) | 11.32 | 8627 | 42067 | 1.039 |  | 100.0% |
| 8 | zipformer-zh-sm-t | clean | 11.83 | 8517 | 41921 | 1.036 |  | 100.0% |
| 9 | zipformer-zh-xl-t | pink(20dB) | 12.10 | 8521 | 42031 | 1.038 |  | 100.0% |
| 10 | zipformer-zh-xl-t | white(20dB) | 12.57 | 8530 | 42038 | 1.039 |  | 100.0% |
| 11 | zipformer-zh-sm | clean | 13.28 | 8381 | 41760 | 1.032 |  | 100.0% |
| 12 | zipformer-zh-sm | pink(20dB) | 13.36 | 8391 | 41790 | 1.032 |  | 100.0% |
| 13 | zipformer-zh-sm-t | pink(20dB) | 13.40 | 8491 | 41994 | 1.037 |  | 100.0% |
| 14 | zipformer-zh-sm-t | white(20dB) | 13.48 | 8489 | 42002 | 1.038 |  | 100.0% |
| 15 | paraformer-tri | pink(20dB) | 13.73 | 8415 | 42208 | 1.043 |  | 100.0% |
| 16 | zipformer-zh-sm | white(20dB) | 13.78 | 8393 | 41799 | 1.033 |  | 100.0% |
| 17 | paraformer-tri | clean | 14.02 | 8495 | 41898 | 1.035 |  | 100.0% |
| 18 | paraformer-tri | white(20dB) | 14.18 | 8427 | 42027 | 1.038 |  | 100.0% |

## Noise Robustness (CER % by Model × Condition)

| Model | clean | pink(20dB) | white(20dB) | CER Δ (clean→worst) |
|-------|-------|-------|-------|---------------------|
| zipformer-zh-xl | 10.70 | 11.11 | 11.32 | +0.62% |
| zipformer-zh-xl-t | 10.93 | 12.10 | 12.57 | +1.64% |
| paraformer | 11.17 | 11.13 | 11.32 | +0.15% |
| zipformer-zh-sm-t | 11.83 | 13.40 | 13.48 | +1.65% |
| zipformer-zh-sm | 13.28 | 13.36 | 13.78 | +0.50% |
| paraformer-tri | 14.02 | 13.73 | 14.18 | +0.16% |

## Per-Model Details

### zipformer-zh-xl — clean

- Runs: 100/100 OK
- CER mean: 10.70% | p50: 10.19%
- TTFP mean: 728 ms
- TTFF mean: 8615 ms
- E2E mean: 42023 ms | p50: 41557 ms
- RTF mean: 1.038 | p50: 1.039
- Avg latency mean:  ms
- P95 latency mean:  ms

### zipformer-zh-xl-t — clean

- Runs: 100/100 OK
- CER mean: 10.93% | p50: 10.57%
- TTFP mean: 800 ms
- TTFF mean: 8559 ms
- E2E mean: 42100 ms | p50: 42017 ms
- RTF mean: 1.040 | p50: 1.039
- Avg latency mean:  ms
- P95 latency mean:  ms

### zipformer-zh-xl — pink(20dB)

- Runs: 100/100 OK
- CER mean: 11.11% | p50: 10.77%
- TTFP mean: 749 ms
- TTFF mean: 8637 ms
- E2E mean: 42056 ms | p50: 41577 ms
- RTF mean: 1.039 | p50: 1.039
- Avg latency mean:  ms
- P95 latency mean:  ms

### paraformer — pink(20dB)

- Runs: 100/100 OK
- CER mean: 11.13% | p50: 10.29%
- TTFP mean: 941 ms
- TTFF mean: 8517 ms
- E2E mean: 41873 ms | p50: 41601 ms
- RTF mean: 1.034 | p50: 1.036
- Avg latency mean:  ms
- P95 latency mean:  ms

### paraformer — clean

- Runs: 100/100 OK
- CER mean: 11.17% | p50: 10.28%
- TTFP mean: 932 ms
- TTFF mean: 8500 ms
- E2E mean: 41841 ms | p50: 41575 ms
- RTF mean: 1.034 | p50: 1.034
- Avg latency mean:  ms
- P95 latency mean:  ms

### paraformer — white(20dB)

- Runs: 100/100 OK
- CER mean: 11.32% | p50: 10.98%
- TTFP mean: 925 ms
- TTFF mean: 8495 ms
- E2E mean: 41897 ms | p50: 41619 ms
- RTF mean: 1.035 | p50: 1.036
- Avg latency mean:  ms
- P95 latency mean:  ms

### zipformer-zh-xl — white(20dB)

- Runs: 100/100 OK
- CER mean: 11.32% | p50: 10.90%
- TTFP mean: 746 ms
- TTFF mean: 8627 ms
- E2E mean: 42067 ms | p50: 41635 ms
- RTF mean: 1.039 | p50: 1.040
- Avg latency mean:  ms
- P95 latency mean:  ms

### zipformer-zh-sm-t — clean

- Runs: 100/100 OK
- CER mean: 11.83% | p50: 11.44%
- TTFP mean: 763 ms
- TTFF mean: 8517 ms
- E2E mean: 41921 ms | p50: 41541 ms
- RTF mean: 1.036 | p50: 1.035
- Avg latency mean:  ms
- P95 latency mean:  ms

### zipformer-zh-xl-t — pink(20dB)

- Runs: 100/100 OK
- CER mean: 12.10% | p50: 11.73%
- TTFP mean: 969 ms
- TTFF mean: 8521 ms
- E2E mean: 42031 ms | p50: 41579 ms
- RTF mean: 1.038 | p50: 1.039
- Avg latency mean:  ms
- P95 latency mean:  ms

### zipformer-zh-xl-t — white(20dB)

- Runs: 100/100 OK
- CER mean: 12.57% | p50: 12.28%
- TTFP mean: 987 ms
- TTFF mean: 8530 ms
- E2E mean: 42038 ms | p50: 41579 ms
- RTF mean: 1.039 | p50: 1.039
- Avg latency mean:  ms
- P95 latency mean:  ms

### zipformer-zh-sm — clean

- Runs: 100/100 OK
- CER mean: 13.28% | p50: 12.83%
- TTFP mean: 681 ms
- TTFF mean: 8381 ms
- E2E mean: 41760 ms | p50: 41218 ms
- RTF mean: 1.032 | p50: 1.032
- Avg latency mean:  ms
- P95 latency mean:  ms

### zipformer-zh-sm — pink(20dB)

- Runs: 100/100 OK
- CER mean: 13.36% | p50: 13.12%
- TTFP mean: 684 ms
- TTFF mean: 8391 ms
- E2E mean: 41790 ms | p50: 41265 ms
- RTF mean: 1.032 | p50: 1.033
- Avg latency mean:  ms
- P95 latency mean:  ms

### zipformer-zh-sm-t — pink(20dB)

- Runs: 100/100 OK
- CER mean: 13.40% | p50: 12.77%
- TTFP mean: 852 ms
- TTFF mean: 8491 ms
- E2E mean: 41994 ms | p50: 41582 ms
- RTF mean: 1.037 | p50: 1.036
- Avg latency mean:  ms
- P95 latency mean:  ms

### zipformer-zh-sm-t — white(20dB)

- Runs: 100/100 OK
- CER mean: 13.48% | p50: 12.96%
- TTFP mean: 855 ms
- TTFF mean: 8489 ms
- E2E mean: 42002 ms | p50: 41590 ms
- RTF mean: 1.038 | p50: 1.037
- Avg latency mean:  ms
- P95 latency mean:  ms

### paraformer-tri — pink(20dB)

- Runs: 100/100 OK
- CER mean: 13.73% | p50: 13.21%
- TTFP mean: 911 ms
- TTFF mean: 8415 ms
- E2E mean: 42208 ms | p50: 42243 ms
- RTF mean: 1.043 | p50: 1.037
- Avg latency mean:  ms
- P95 latency mean:  ms

### zipformer-zh-sm — white(20dB)

- Runs: 100/100 OK
- CER mean: 13.78% | p50: 13.57%
- TTFP mean: 683 ms
- TTFF mean: 8393 ms
- E2E mean: 41799 ms | p50: 41269 ms
- RTF mean: 1.033 | p50: 1.033
- Avg latency mean:  ms
- P95 latency mean:  ms

### paraformer-tri — clean

- Runs: 100/100 OK
- CER mean: 14.02% | p50: 13.31%
- TTFP mean: 934 ms
- TTFF mean: 8495 ms
- E2E mean: 41898 ms | p50: 41575 ms
- RTF mean: 1.035 | p50: 1.035
- Avg latency mean:  ms
- P95 latency mean:  ms

### paraformer-tri — white(20dB)

- Runs: 100/100 OK
- CER mean: 14.18% | p50: 13.81%
- TTFP mean: 909 ms
- TTFF mean: 8427 ms
- E2E mean: 42027 ms | p50: 41588 ms
- RTF mean: 1.038 | p50: 1.035
- Avg latency mean:  ms
- P95 latency mean:  ms

## Worst CER Cases (top 10)

| Model | Noise | Case | CER (%) | Ref | Hyp |
|-------|-------|------|---------|-----|-----|
| zipformer-zh-sm-t | pink(20dB) | d0014 | 24.36 | 您好，我有一張凱基人壽活利鑫動變額壽險想詢問一些問題。我想了 | 明浩我有一張凱奇人忒力心動變額妾想詢問一些問題瞭解一下目前我 |
| zipformer-zh-xl-t | white(20dB) | d0046 | 24.31 | 您好，我的凱基人壽心康泰住院醫療限額給付健康保險附約之前因為 | 名號我的凱基人忻康泰住院醫療限額幾部健康保險赴約之前因為忘記 |
| zipformer-zh-sm-t | white(20dB) | d0046 | 23.20 | 您好，我的凱基人壽心康泰住院醫療限額給付健康保險附約之前因為 | 明浩我的寇基人壽星康泰住院醫療限額給部健康保險赴約之前因為旺 |
| zipformer-zh-sm | white(20dB) | d0046 | 22.65 | 您好，我的凱基人壽心康泰住院醫療限額給付健康保險附約之前因為 | 您號我的凱基人新康泰住院醫療限額幾部健康保險赴約之前因為忘記 |
| paraformer-tri | white(20dB) | d0046 | 22.10 | 您好，我的凱基人壽心康泰住院醫療限額給付健康保險附約之前因為 | 您好我的凱基元壽新康泰住院醫療限額及部健康赴險赴約之前因為忘 |
| zipformer-zh-sm | white(20dB) | d0046 | 22.10 | 您好，我的凱基人壽心康泰住院醫療限額給付健康保險附約之前因為 | 您好我的凱基人新康泰住院醫療限額幾部健康保險赴約之前因為忘記 |
| zipformer-zh-sm | pink(20dB) | d0046 | 22.10 | 您好，我的凱基人壽心康泰住院醫療限額給付健康保險附約之前因為 | 名號我的凱基元新康泰住院醫療限額幾負健康保險赴約之前因為忘記 |
| zipformer-zh-sm | white(20dB) | d0032 | 21.60 | 您好，我想了解凱基人壽永享樂活住院醫療定期保險的保障細節。我 | 您好我想了解凱極忍受影響樂活住院醫療並其保險的保障細節我想了 |
| zipformer-zh-xl-t | white(20dB) | d0046 | 21.55 | 您好，我的凱基人壽心康泰住院醫療限額給付健康保險附約之前因為 | 您好我的凱基人忻康泰住院醫療限額及部健康保險赴約之前因為忘記 |
| zipformer-zh-sm-t | pink(20dB) | d0023 | 21.47 | 您好，我想詢問凱基人壽心安心照護終身保險辦理減額繳清的相關事 | 您好我想袁機人忻安心賬戶終身保險辦理減額繳慶的相關事宜我最近 |
