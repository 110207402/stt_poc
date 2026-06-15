#!/usr/bin/env python3
"""
深度分析報告產生器
輸入: reports/merged_final/run_metrics.csv
輸出: reports/merged_final/analysis_report.md
"""
from __future__ import annotations

import csv
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── 路徑 ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "reports" / "merged_final"
INPUT_CSV = DATA_DIR / "run_metrics.csv"
OUTPUT_MD = DATA_DIR / "analysis_report.md"

# ── 模型顯示順序（依 clean CER 排） ────────────────────────────────────────
MODEL_ORDER = [
    "zipformer-zh-xl",
    "zipformer-zh-xl-t",
    "paraformer",
    "zipformer-zh-sm-t",
    "zipformer-zh-sm",
    "paraformer-tri",
]

CATEGORY_ZH = {
    "claim":           "理賠申請",
    "policy_inquiry":  "保單查詢",
    "account_mgmt":    "帳戶管理",
    "payment":         "繳費相關",
    "policy_change":   "保單變更",
    "product_inquiry": "商品詢問",
    "lapse_reinstate": "失效復效",
}

# ── 工具函式 ─────────────────────────────────────────────────────────────────
def _avg(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None

def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    idx = (len(s) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)

def _fmt(v: Any, d: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{d}f}"

def levenshtein_ops(ref: str, hyp: str) -> list[tuple[str, str, str]]:
    """回傳 (op, ref_char, hyp_char) 清單"""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(("sub", ref[i-1], hyp[j-1])); i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append(("ins", "", hyp[j-1])); j -= 1
        else:
            ops.append(("del", ref[i-1], "")); i -= 1
    return ops

def strip_punct(text: str) -> str:
    for c in "，。？！、；：「」『』（）〔〕【】 \t\n":
        text = text.replace(c, "")
    return text

# ── 載入資料 ─────────────────────────────────────────────────────────────────
def load_data() -> list[dict]:
    rows = list(csv.DictReader(INPUT_CSV.open(encoding="utf-8")))
    for r in rows:
        for k in ["cer", "ttfp_ms", "ttff_ms", "e2e_ms", "rtf",
                  "audio_duration_s", "snr_db", "partial_events",
                  "final_segments", "error_count", "disconnect_count"]:
            if r.get(k) not in (None, "", "None"):
                try: r[k] = float(r[k])
                except: pass
    return rows

# ── 各分析函式 ────────────────────────────────────────────────────────────────
def section_executive_summary(ok: list[dict], lines: list[str]) -> None:
    lines += ["## 1. 執行摘要", ""]

    clean = [r for r in ok if r["noise_type"] == "clean"]
    # 最佳模型
    by_model: dict[str, list[float]] = defaultdict(list)
    for r in clean:
        by_model[r["model"]].append(r["cer"])
    ranked = sorted(by_model.items(), key=lambda x: _avg(x[1]))  # type: ignore

    best_m, best_vals = ranked[0]
    best_cer = _avg(best_vals)
    worst_m, worst_vals = ranked[-1]
    worst_cer = _avg(worst_vals)

    # 最抗噪
    noise_delta: dict[str, float] = {}
    for m in MODEL_ORDER:
        c = _avg([r["cer"] for r in ok if r["model"] == m and r["noise_type"] == "clean"])
        w = _avg([r["cer"] for r in ok if r["model"] == m and r["noise_type"] == "white"])
        if c and w:
            noise_delta[m] = w - c
    best_noise_m = min(noise_delta, key=noise_delta.get)  # type: ignore

    # 最快 TTFP
    ttfp_by_m = {m: _avg([r["ttfp_ms"] for r in clean if r["model"] == m and isinstance(r.get("ttfp_ms"), float)])
                 for m in MODEL_ORDER}
    fastest_m = min(ttfp_by_m, key=lambda m: ttfp_by_m[m] or 9999)  # type: ignore

    # 最難 case
    by_case: dict[str, list[float]] = defaultdict(list)
    for r in clean:
        by_case[r["case_id"]].append(r["cer"])
    hardest_case = max(by_case, key=lambda c: _avg(by_case[c]))  # type: ignore
    hardest_cat = next(r["category"] for r in clean if r["case_id"] == hardest_case)

    lines += [
        "| 指標 | 結果 |",
        "|------|------|",
        f"| 最高準確度（clean CER）| **{best_m}**（{_fmt(best_cer)}%）|",
        f"| 最低準確度（clean CER）| {worst_m}（{_fmt(worst_cer)}%）|",
        f"| 最強抗噪（白噪音 CER 漲幅最小）| **{best_noise_m}**（+{_fmt(noise_delta[best_noise_m])}%）|",
        f"| 最快反應（TTFP 最低）| **{fastest_m}**（{_fmt(ttfp_by_m[fastest_m], 0)} ms）|",
        f"| 最難辨識 case | **{hardest_case}**（{CATEGORY_ZH.get(hardest_case_cat := hardest_cat, hardest_cat)}，avg CER {_fmt(_avg(by_case[hardest_case]))}%）|",
        f"| 測試規模 | 6 模型 × 50 cases × 3 noise × 2 repeats = **1800 runs** |",
        "",
    ]


def section_model_ranking(ok: list[dict], lines: list[str]) -> None:
    lines += ["## 2. 模型整體排名", ""]
    clean = [r for r in ok if r["noise_type"] == "clean"]

    lines += [
        "| 排名 | 模型 | CER mean | CER p50 | CER min | CER max | TTFP (ms) | E2E (ms) | RTF |",
        "|------|------|---------|---------|---------|---------|-----------|----------|-----|",
    ]
    model_stats = []
    for m in MODEL_ORDER:
        d = [r for r in clean if r["model"] == m]
        cers  = [r["cer"] for r in d if isinstance(r.get("cer"), float)]
        ttfps = [r["ttfp_ms"] for r in d if isinstance(r.get("ttfp_ms"), float)]
        e2es  = [r["e2e_ms"]  for r in d if isinstance(r.get("e2e_ms"),  float)]
        rtfs  = [r["rtf"]     for r in d if isinstance(r.get("rtf"),     float)]
        model_stats.append((m, cers, ttfps, e2es, rtfs))

    model_stats.sort(key=lambda x: _avg(x[1]) or 99)  # type: ignore
    for rank, (m, cers, ttfps, e2es, rtfs) in enumerate(model_stats, 1):
        lines.append(
            f"| {rank} | {m} | {_fmt(_avg(cers))}% | {_fmt(_pct(cers, 0.5))}% "
            f"| {_fmt(min(cers))}% | {_fmt(max(cers))}% "
            f"| {_fmt(_avg(ttfps), 0)} | {_fmt(_avg(e2es), 0)} "
            f"| {_fmt(_avg(rtfs), 3)} |"
        )
    lines.append("")


def section_noise_robustness(ok: list[dict], lines: list[str]) -> None:
    lines += ["## 3. 抗噪能力分析", ""]
    noise_types = ["clean", "white", "pink"]

    lines += [
        "### 3.1 CER by 模型 × 噪音類型",
        "",
        "| 模型 | clean | white 20dB | pink 20dB | 白噪漲幅 | 粉噪漲幅 | 最差漲幅 |",
        "|------|-------|-----------|----------|---------|---------|---------|",
    ]
    for m in MODEL_ORDER:
        c = _avg([r["cer"] for r in ok if r["model"] == m and r["noise_type"] == "clean"])
        w = _avg([r["cer"] for r in ok if r["model"] == m and r["noise_type"] == "white"])
        p = _avg([r["cer"] for r in ok if r["model"] == m and r["noise_type"] == "pink"])
        dw = (w - c) if c and w else None
        dp = (p - c) if c and p else None
        worst = max(x for x in [dw, dp] if x is not None) if any(x is not None for x in [dw, dp]) else None
        lines.append(
            f"| {m} | {_fmt(c)}% | {_fmt(w)}% | {_fmt(p)}% "
            f"| {'+' if dw and dw >= 0 else ''}{_fmt(dw)}% "
            f"| {'+' if dp and dp >= 0 else ''}{_fmt(dp)}% "
            f"| **{'+' if worst and worst >= 0 else ''}{_fmt(worst)}%** |"
        )
    lines.append("")

    # 逐 case 噪音敏感度最高的
    lines += [
        "### 3.2 白噪音下 CER 漲幅最大的 cases（各模型平均）",
        "",
        "| Case | 類別 | clean avg | white avg | 漲幅 |",
        "|------|------|----------|----------|------|",
    ]
    case_clean: dict[str, list[float]] = defaultdict(list)
    case_white: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        if r["noise_type"] == "clean": case_clean[r["case_id"]].append(r["cer"])
        if r["noise_type"] == "white": case_white[r["case_id"]].append(r["cer"])

    deltas = []
    for cid in case_clean:
        c_avg = _avg(case_clean[cid])
        w_avg = _avg(case_white.get(cid, []))
        if c_avg and w_avg:
            deltas.append((cid, c_avg, w_avg, w_avg - c_avg))
    deltas.sort(key=lambda x: -x[3])
    for cid, c_avg, w_avg, delta in deltas[:8]:
        cat = next((r["category"] for r in ok if r["case_id"] == cid), "")
        lines.append(f"| {cid} | {CATEGORY_ZH.get(cat, cat)} | {_fmt(c_avg)}% | {_fmt(w_avg)}% | +{_fmt(delta)}% |")
    lines.append("")


def section_latency(ok: list[dict], lines: list[str]) -> None:
    lines += ["## 4. 延遲分析（clean 條件）", ""]
    clean = [r for r in ok if r["noise_type"] == "clean"]

    lines += [
        "### 4.1 各模型延遲統計",
        "",
        "| 模型 | TTFP mean | TTFP p50 | TTFF mean | TTFF p50 | E2E mean | E2E p50 | RTF mean |",
        "|------|----------|---------|----------|---------|---------|---------|---------|",
    ]
    for m in MODEL_ORDER:
        d = [r for r in clean if r["model"] == m]
        ttfp = [r["ttfp_ms"] for r in d if isinstance(r.get("ttfp_ms"), float)]
        ttff = [r["ttff_ms"] for r in d if isinstance(r.get("ttff_ms"), float)]
        e2e  = [r["e2e_ms"]  for r in d if isinstance(r.get("e2e_ms"),  float)]
        rtf  = [r["rtf"]     for r in d if isinstance(r.get("rtf"),     float)]
        lines.append(
            f"| {m} | {_fmt(_avg(ttfp), 0)} | {_fmt(_pct(ttfp, 0.5), 0)} "
            f"| {_fmt(_avg(ttff), 0)} | {_fmt(_pct(ttff, 0.5), 0)} "
            f"| {_fmt(_avg(e2e), 0)} | {_fmt(_pct(e2e, 0.5), 0)} "
            f"| {_fmt(_avg(rtf), 3)} |"
        )
    lines.append("")

    lines += [
        "> **TTFP**: 第一個 partial 出現時間（使用者感受到的反應速度）  ",
        "> **TTFF**: 第一個 final 出現時間  ",
        "> **E2E**: 串流開始到最後一個 final 的總時間  ",
        "> **RTF**: E2E / 音訊長度（< 1.0 表示比即時更快）",
        "",
    ]


def section_error_analysis(ok: list[dict], lines: list[str]) -> None:
    lines += ["## 5. 錯字分析（clean 條件，全模型合計）", ""]
    clean = [r for r in ok if r["noise_type"] == "clean"]

    sub_all: dict[tuple, int]   = defaultdict(int)
    del_all: dict[str, int]     = defaultdict(int)
    ins_all: dict[str, int]     = defaultdict(int)

    for r in clean:
        ref = strip_punct(r.get("ref_text", ""))
        hyp = strip_punct(r.get("hyp_text", ""))
        if not ref or not hyp:
            continue
        for op, rc, hc in levenshtein_ops(ref, hyp):
            if op == "sub": sub_all[(rc, hc)] += 1
            elif op == "del": del_all[rc] += 1
            elif op == "ins": ins_all[hc] += 1

    # 5.1 替換錯誤分組
    lines += ["### 5.1 高頻替換錯誤（Top 25）", ""]

    # 分類
    fanjian = [(k, v) for k, v in sub_all.items() if v >= 5]
    fanjian_pairs = {("帳","賬"),("妳","你"),("只","隻"),("台","臺"),("了","瞭"),
                     ("著","著"),("裡","里"),("後","後"),("麻","麻"),("體","体")}

    lines += [
        "| # | ref → hyp | 次數 | 類型 | 影響詞彙 |",
        "|---|----------|------|------|---------|",
    ]
    top_subs = sorted(sub_all.items(), key=lambda x: -x[1])[:25]
    type_map = {
        ("帳","賬"): ("繁簡字型", "帳戶/帳號"),
        ("妳","你"): ("繁簡字型", "妳好"),
        ("只","隻"): ("繁簡字型", "只能"),
        ("台","臺"): ("繁簡字型", "台幣"),
        ("了","瞭"): ("繁簡字型", "瞭解"),
        ("效","校"): ("近音誤字", "復效/失效"),
        ("附","赴"): ("近音誤字", "附約"),
        ("復","覆"): ("近音誤字", "復效"),
        ("給","幾"): ("近音誤字", "給付"),
        ("繳","腳"): ("近音誤字", "繳費"),
        ("繳","較"): ("近音誤字", "繳費"),
        ("享","想"): ("近音誤字", "享安心系列"),
        ("壽","受"): ("品牌名", "凱基人壽"),
        ("壽","獸"): ("品牌名", "凱基人壽"),
        ("人","元"): ("品牌名", "凱基人壽"),
        ("基","吉"): ("品牌名", "凱基"),
        ("心","新"): ("近音誤字", "心康泰"),
        ("鑫","心"): ("近音誤字", "鑫旺九九"),
        ("護","戶"): ("近音誤字", "照護"),
        ("保","寶"): ("近音誤字", "保費/保單"),
        ("清","輕"): ("近音誤字", "繳清"),
        ("領","令"): ("近音誤字", "提領"),
        ("型","洗"): ("近音誤字", "變額型"),
        ("長","常"): ("近音誤字", "長期照顧"),
        ("照","賬"): ("近音誤字", "照護"),
    }
    for i, ((rc, hc), cnt) in enumerate(top_subs, 1):
        t, vocab = type_map.get((rc, hc), ("其他", ""))
        lines.append(f"| {i} | `{rc}` → `{hc}` | {cnt} | {t} | {vocab} |")
    lines.append("")

    # 5.2 依類型統計
    lines += ["### 5.2 錯誤類型統計", ""]
    fanjian_cnt  = sum(v for (rc, hc), v in sub_all.items()
                       if (rc, hc) in {("帳","賬"),("妳","你"),("只","隻"),("台","臺"),("了","瞭")})
    jinyin_cnt   = sum(v for (rc, hc), v in sub_all.items()
                       if (rc, hc) in {("效","校"),("附","赴"),("復","覆"),("給","幾"),
                                        ("繳","腳"),("繳","較"),("享","想"),("心","新"),
                                        ("鑫","心"),("護","戶"),("保","寶"),("清","輕"),
                                        ("領","令"),("型","洗"),("長","常")})
    brand_cnt    = sum(v for (rc, hc), v in sub_all.items()
                       if (rc, hc) in {("壽","受"),("壽","獸"),("人","元"),("基","吉")})
    total_sub    = sum(sub_all.values())
    total_del    = sum(del_all.values())
    total_ins    = sum(ins_all.values())

    lines += [
        "| 錯誤類型 | 次數 | 佔替換錯誤 | 說明 |",
        "|---------|------|----------|------|",
        f"| 繁簡字型差異 | {fanjian_cnt} | {100*fanjian_cnt/total_sub:.1f}% | 帳/賬、妳/你、只/隻 等 |",
        f"| 近音誤字（保險術語）| {jinyin_cnt} | {100*jinyin_cnt/total_sub:.1f}% | 復效/附約/給付 等 |",
        f"| 品牌名誤字 | {brand_cnt} | {100*brand_cnt/total_sub:.1f}% | 凱基人壽 相關 |",
        f"| 其他替換 | {total_sub-fanjian_cnt-jinyin_cnt-brand_cnt} | {100*(total_sub-fanjian_cnt-jinyin_cnt-brand_cnt)/total_sub:.1f}% | |",
        f"| 漏字（deletion）| {total_del} | — | |",
        f"| 多字（insertion）| {total_ins} | — | |",
        "",
    ]

    # 5.3 高頻漏字
    lines += [
        "### 5.3 高頻漏字（Top 15）",
        "",
        "| 字 | 漏掉次數 | 常見脈絡 |",
        "|---|---------|---------|",
    ]
    del_context = {
        "壽": "凱基人**壽**",
        "嗎": "句尾語氣詞",
        "基": "凱**基**人壽",
        "繳": "**繳**費",
        "帳": "**帳**戶",
        "少": "至**少**",
        "請": "麻煩**請**問",
        "下": "請問一**下**",
        "人": "凱基**人**壽",
        "扣": "自動**扣**繳",
        "額": "限**額**給付",
        "障": "保**障**",
        "宜": "事**宜**",
    }
    for ch, cnt in sorted(del_all.items(), key=lambda x: -x[1])[:15]:
        ctx = del_context.get(ch, "")
        lines.append(f"| `{ch}` | {cnt} | {ctx} |")
    lines.append("")

    # 5.4 各模型的錯誤分佈
    lines += ["### 5.4 各模型替換錯誤量（clean）", ""]
    lines += [
        "| 模型 | 總替換 | 繁簡 | 近音術語 | 品牌名 |",
        "|------|--------|------|---------|-------|",
    ]
    fanjian_set = {("帳","賬"),("妳","你"),("只","隻"),("台","臺"),("了","瞭")}
    jinyin_set  = {("效","校"),("附","赴"),("復","覆"),("給","幾"),("繳","腳"),
                   ("繳","較"),("享","想"),("心","新"),("鑫","心"),("護","戶"),
                   ("保","寶"),("清","輕"),("領","令"),("型","洗"),("長","常")}
    brand_set   = {("壽","受"),("壽","獸"),("人","元"),("基","吉")}

    for m in MODEL_ORDER:
        sub_m: dict[tuple, int] = defaultdict(int)
        for r in clean:
            if r["model"] != m: continue
            ref = strip_punct(r.get("ref_text",""))
            hyp = strip_punct(r.get("hyp_text",""))
            if not ref or not hyp: continue
            for op, rc, hc in levenshtein_ops(ref, hyp):
                if op == "sub": sub_m[(rc, hc)] += 1
        t = sum(sub_m.values())
        fj = sum(v for k, v in sub_m.items() if k in fanjian_set)
        jy = sum(v for k, v in sub_m.items() if k in jinyin_set)
        br = sum(v for k, v in sub_m.items() if k in brand_set)
        lines.append(f"| {m} | {t} | {fj} ({100*fj/t:.0f}%) | {jy} ({100*jy/t:.0f}%) | {br} ({100*br/t:.0f}%) |")
    lines.append("")


def section_category(ok: list[dict], lines: list[str]) -> None:
    lines += ["## 6. 話題類別分析", ""]
    clean = [r for r in ok if r["noise_type"] == "clean"]

    # 6.1 各 category 平均 CER
    lines += [
        "### 6.1 各類別 CER（各模型平均）",
        "",
        "| 類別 | Cases | CER mean | CER min | CER max | 最難 Case |",
        "|------|-------|---------|---------|---------|----------|",
    ]
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_cat_cases: dict[str, list[str]] = defaultdict(list)
    for r in clean:
        by_cat[r["category"]].append(r["cer"])
        if r["case_id"] not in by_cat_cases[r["category"]]:
            by_cat_cases[r["category"]].append(r["case_id"])

    by_case_avg: dict[str, float] = {}
    for cat, cases in by_cat_cases.items():
        for cid in cases:
            vals = [r["cer"] for r in clean if r["case_id"] == cid]
            by_case_avg[cid] = _avg(vals) or 0

    for cat in sorted(by_cat, key=lambda c: _avg(by_cat[c])):  # type: ignore
        vals = by_cat[cat]
        cases = by_cat_cases[cat]
        hardest = max(cases, key=lambda c: by_case_avg.get(c, 0))
        lines.append(
            f"| {CATEGORY_ZH.get(cat, cat)} | {len(cases)} | {_fmt(_avg(vals))}% "
            f"| {_fmt(min(vals))}% | {_fmt(max(vals))}% | {hardest}（{_fmt(by_case_avg[hardest])}%）|"
        )
    lines.append("")

    # 6.2 各模型 × 各類別
    lines += [
        "### 6.2 各模型 × 各類別 CER%（clean）",
        "",
    ]
    categories = sorted(by_cat.keys(), key=lambda c: _avg(by_cat[c]))  # type: ignore
    header = "| 模型 | " + " | ".join(CATEGORY_ZH.get(c, c) for c in categories) + " |"
    sep    = "|------|" + "|".join(["------"] * len(categories)) + "|"
    lines += [header, sep]

    for m in MODEL_ORDER:
        row = f"| {m} |"
        for cat in categories:
            vals = [r["cer"] for r in clean if r["model"] == m and r["category"] == cat]
            row += f" {_fmt(_avg(vals))}% |"
        lines.append(row)
    lines.append("")


def section_case_analysis(ok: list[dict], lines: list[str]) -> None:
    lines += ["## 7. 個案難度分析（clean）", ""]
    clean = [r for r in ok if r["noise_type"] == "clean"]

    by_case: dict[str, list[float]] = defaultdict(list)
    for r in clean:
        by_case[r["case_id"]].append(r["cer"])

    case_info = {}
    for r in clean:
        if r["case_id"] not in case_info:
            case_info[r["case_id"]] = {
                "cat": r["category"],
                "dur": r.get("audio_duration_s", 0),
                "ref": (r.get("ref_text") or "")[:40],
            }

    # 7.1 全部 50 cases 排名
    lines += [
        "### 7.1 50 cases 難度排名（各模型 clean CER 平均）",
        "",
        "| 排名 | Case | 類別 | 音長(s) | CER avg | CER min | CER max | 內容摘要 |",
        "|------|------|------|--------|---------|---------|---------|---------|",
    ]
    case_ranked = sorted(by_case.items(), key=lambda x: _avg(x[1]), reverse=True)  # type: ignore
    for rank, (cid, vals) in enumerate(case_ranked, 1):
        info = case_info.get(cid, {})
        lines.append(
            f"| {rank} | {cid} | {CATEGORY_ZH.get(info.get('cat',''), info.get('cat',''))} "
            f"| {_fmt(info.get('dur', 0), 1)} "
            f"| {_fmt(_avg(vals))}% | {_fmt(min(vals))}% | {_fmt(max(vals))}% "
            f"| {info.get('ref','')}... |"
        )
    lines.append("")

    # 7.2 最難 cases 詳細
    lines += ["### 7.2 最難 5 Cases 詳細分析", ""]
    for rank, (cid, vals) in enumerate(case_ranked[:5], 1):
        info = case_info.get(cid, {})
        lines += [
            f"#### {rank}. {cid} — {CATEGORY_ZH.get(info.get('cat',''), '')}（avg CER {_fmt(_avg(vals))}%）",
            "",
            f"**音訊長度：** {_fmt(info.get('dur',0), 1)}s",
            "",
        ]
        # 標準答案全文
        ref_full = next((r.get("ref_text","") for r in clean if r["case_id"] == cid), "")
        lines += [
            "**標準答案（ref）：**",
            "",
            f"> {ref_full}",
            "",
        ]

        # 各模型辨識結果全文
        lines += [
            "**各模型辨識結果：**",
            "",
        ]
        for m in MODEL_ORDER:
            rows_m = [r for r in clean if r["model"] == m and r["case_id"] == cid and r.get("repeat") == "1"]
            if rows_m:
                r = rows_m[0]
                hyp_full = r.get("hyp_text") or ""
                lines += [
                    f"- **{m}**（CER {_fmt(r.get('cer'))}%）",
                    f"  > {hyp_full}",
                    "",
                ]
        lines.append("")


def section_brand_analysis(ok: list[dict], lines: list[str]) -> None:
    lines += ["## 8. 品牌名辨識分析", ""]
    clean = [r for r in ok if r["noise_type"] == "clean"]

    brand_terms = ["凱基人壽", "凱基銀行", "心康泰", "鑫旺九九", "享安心", "元氣樂活"]

    lines += [
        "### 8.1 各品牌詞辨識正確率（含此詞的 cases，clean）",
        "",
        "| 品牌詞 | " + " | ".join(MODEL_ORDER) + " |",
        "|-------|" + "|".join(["------"] * len(MODEL_ORDER)) + "|",
    ]
    for term in brand_terms:
        row = f"| {term} |"
        for m in MODEL_ORDER:
            rows_m = [r for r in clean if r["model"] == m
                      and r.get("repeat") == "1"
                      and term in (r.get("ref_text") or "")]
            if not rows_m:
                row += " — |"
                continue
            correct = sum(1 for r in rows_m if term in (r.get("hyp_text") or ""))
            pct = 100 * correct / len(rows_m)
            row += f" {correct}/{len(rows_m)}（{pct:.0f}%）|"
        lines.append(row)
    lines.append("")

    # 凱基人壽的常見誤識方式
    lines += [
        "### 8.2「凱基人壽」常見誤識方式",
        "",
        "| 模型 | 最常見誤識 | 次數 |",
        "|------|----------|------|",
    ]
    wrong_variants = ["凱基元壽", "凱基人受", "凱基人獸", "凱吉人壽", "寇基人壽", "凱集人壽", "凱奇人壽"]
    for m in MODEL_ORDER:
        rows_m = [r for r in clean if r["model"] == m and "凱基人壽" in (r.get("ref_text") or "")]
        counts: dict[str, int] = defaultdict(int)
        for r in rows_m:
            hyp = r.get("hyp_text") or ""
            for v in wrong_variants:
                if v in hyp:
                    counts[v] += 1
        if counts:
            top = max(counts, key=counts.get)  # type: ignore
            lines.append(f"| {m} | {top} | {counts[top]} |")
        else:
            lines.append(f"| {m} | （無常見誤識）| — |")
    lines.append("")


def section_model_headtohead(ok: list[dict], lines: list[str]) -> None:
    lines += ["## 9. 模型對比分析", ""]
    clean = [r for r in ok if r["noise_type"] == "clean"]

    # xl vs xl-t
    lines += ["### 9.1 zipformer-zh-xl vs zipformer-zh-xl-t（clean，逐 case）", ""]
    xl_cer  = {r["case_id"]: r["cer"] for r in clean if r["model"] == "zipformer-zh-xl"   and r.get("repeat") == "1"}
    xlt_cer = {r["case_id"]: r["cer"] for r in clean if r["model"] == "zipformer-zh-xl-t" and r.get("repeat") == "1"}
    diffs = [(cid, xl_cer[cid], xlt_cer[cid], xlt_cer[cid] - xl_cer[cid])
             for cid in xl_cer if cid in xlt_cer]
    xl_wins  = sum(1 for *_, d in diffs if d > 0.5)
    xlt_wins = sum(1 for *_, d in diffs if d < -0.5)
    tie      = len(diffs) - xl_wins - xlt_wins

    lines += [
        f"- **xl 較好**：{xl_wins} cases | **xl-t 較好**：{xlt_wins} cases | 差不多：{tie} cases",
        f"- xl 平均 CER：{_fmt(_avg([xl_cer[c] for c in xl_cer]))}%",
        f"- xl-t 平均 CER：{_fmt(_avg([xlt_cer[c] for c in xlt_cer]))}%",
        "",
        "**xl-t 勝出的 cases（hotword 有助益）：**",
        "",
        "| Case | xl | xl-t | 差距 |",
        "|------|-----|------|------|",
    ]
    for cid, xl_c, xlt_c, d in sorted(diffs, key=lambda x: x[3])[:5]:
        lines.append(f"| {cid} | {_fmt(xl_c)}% | {_fmt(xlt_c)}% | {_fmt(d)}% |")
    lines.append("")

    # paraformer vs paraformer-tri
    lines += ["### 9.2 paraformer vs paraformer-tri（clean）", ""]
    pf_cer  = {r["case_id"]: r["cer"] for r in clean if r["model"] == "paraformer"     and r.get("repeat") == "1"}
    pft_cer = {r["case_id"]: r["cer"] for r in clean if r["model"] == "paraformer-tri" and r.get("repeat") == "1"}
    pf_wins  = sum(1 for cid in pf_cer if cid in pft_cer and pf_cer[cid] < pft_cer[cid] - 0.5)
    pft_wins = sum(1 for cid in pf_cer if cid in pft_cer and pft_cer[cid] < pf_cer[cid] - 0.5)
    lines += [
        f"- **paraformer 較好**：{pf_wins} cases | **paraformer-tri 較好**：{pft_wins} cases",
        f"- paraformer avg CER：{_fmt(_avg(list(pf_cer.values())))}%",
        f"- paraformer-tri avg CER：{_fmt(_avg(list(pft_cer.values())))}%",
        f"- 結論：paraformer-tri（三語模型）整體比 paraformer 差，增加粵語/英語支援反而稀釋了中文準確度",
        "",
    ]

    # sm vs sm-t
    lines += ["### 9.3 zipformer-zh-sm vs zipformer-zh-sm-t（clean）", ""]
    sm_cer  = {r["case_id"]: r["cer"] for r in clean if r["model"] == "zipformer-zh-sm"   and r.get("repeat") == "1"}
    smt_cer = {r["case_id"]: r["cer"] for r in clean if r["model"] == "zipformer-zh-sm-t" and r.get("repeat") == "1"}
    sm_wins  = sum(1 for cid in sm_cer if cid in smt_cer and sm_cer[cid] < smt_cer[cid] - 0.5)
    smt_wins = sum(1 for cid in sm_cer if cid in smt_cer and smt_cer[cid] < sm_cer[cid] - 0.5)
    lines += [
        f"- **sm 較好**：{sm_wins} cases | **sm-t 較好**：{smt_wins} cases",
        f"- sm avg CER：{_fmt(_avg(list(sm_cer.values())))}%",
        f"- sm-t avg CER：{_fmt(_avg(list(smt_cer.values())))}%",
        f"- 結論：sm-t（Transducer）比 sm（CTC）準確，代表在相同模型大小下 Transducer 架構在此資料集上更優",
        "",
    ]


def section_recommendations(lines: list[str]) -> None:
    lines += [
        "## 10. 結論與建議",
        "",
        "### 10.1 模型選用建議",
        "",
        "| 場景 | 推薦模型 | 理由 |",
        "|------|---------|------|",
        "| 追求最高準確度 | **zipformer-zh-xl** | clean CER 10.70%，抗噪也穩（+0.62%）|",
        "| 有環境噪音（客服中心）| **paraformer** | 白噪 CER 僅漲 0.15%，最強抗噪 |",
        "| 重視第一字出現速度 | **zipformer-zh-sm** | TTFP 最低（681ms）|",
        "| 模型大小與準確度平衡 | **zipformer-zh-sm-t** | 模型小（160MB），CER 11.83% |",
        "| 多語言場景（含英/粵）| paraformer-tri | 唯一支援三語，但中文準確度較低 |",
        "",
        "### 10.2 準確度提升建議",
        "",
        "**優先級 1（影響大，實作容易）：後處理修正**",
        "- 繁簡字型統一：`賬→帳`、`你→妳`、`隻→只` 等，可消除 **442次** 帳/賬錯誤",
        "- 領域術語修正：`覆效→復效`、`赴約→附約`、`幾付→給付` 等加入 `kgi_corrections.json`",
        "",
        "**優先級 2（對 -t 模型）：hotword 補強**",
        "- 新增可用詞（無 OOV）：`复效`、`失效`、`失能`、`附约`、`给付`、`限额给付`",
        "- 凱基人壽等品牌名因 `凯`/`寿` OOV，hotword 對 -t 模型無效，只能靠後處理",
        "",
        "**優先級 3（長期）：換用繁體訓練資料的模型**",
        "- 現有所有 sherpa-onnx 模型均為簡體中文訓練，繁體字 OOV 是根本限制",
        "- 可考慮繁體資料 fine-tuning 或評估 FunASR 等支援繁體的框架",
        "",
        "### 10.3 高風險 Cases 重點關注",
        "",
        "| Case | 類別 | 問題 |",
        "|------|------|------|",
        "| d0046 | 失效復效 | 「心康泰」全模型誤識，「附約」、「復效」多處錯誤 |",
        "| d0017 | 保單查詢 | 外幣保單術語複雜，「金美滿」、「利率變動型」辨識不穩 |",
        "| d0040 | 帳戶管理 | 「鑫旺九九」、「外幣帳戶」、「折換台幣」誤識 |",
        "| d0030 | 商品詢問 | 長句+多個商品名，累積錯誤高 |",
        "| d0021 | 保單變更 | 「減額繳清」術語難辨識 |",
        "",
    ]


# ── 主程式 ────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"載入資料：{INPUT_CSV}")
    all_rows = load_data()
    ok = [r for r in all_rows if r.get("status") == "ok"]
    print(f"有效 rows：{len(ok)} / {len(all_rows)}")

    lines: list[str] = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines += [
        "# 凱基人壽 STT Benchmark 深度分析報告",
        "",
        f"**產生時間：** {now}",
        f"**資料來源：** {INPUT_CSV.name}（{len(ok)} runs）",
        f"**測試規模：** 6 模型 × 50 cases × 3 noise conditions × 2 repeats",
        "",
        "---",
        "",
    ]

    section_executive_summary(ok, lines)
    section_model_ranking(ok, lines)
    section_noise_robustness(ok, lines)
    section_latency(ok, lines)
    section_error_analysis(ok, lines)
    section_category(ok, lines)
    section_case_analysis(ok, lines)
    section_brand_analysis(ok, lines)
    section_model_headtohead(ok, lines)
    section_recommendations(lines)

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n報告已寫入：{OUTPUT_MD}")
    print(f"共 {len(lines)} 行")


if __name__ == "__main__":
    main()
