"""
Phase 3 ASR evaluation — comprehensive analysis of Breeze-ASR-26 4-way ablation.

Reads:
  results/phase3_breeze_asr_results.csv  (case_id, ref, hyp_e1..e4, language, subtype, dur_sec)
  data/phase3_queries.csv                (pii_types, pii_values, domain_terms)

Writes:
  results/phase3_asr_eval_per_query.csv  (one row per query × ablation, all metrics)
  results/phase3_asr_eval_summary.csv    (aggregated tables, long format)
  results/phase3_asr_eval_report.md      (human-readable analysis)

Metrics computed:
  - CER (character error rate)            : strict (punct stripped)
  - CER_norm                                : after CN ↔ Arabic digit normalization
  - Domain term recall                     : did each insurance term land in HYP?
  - PII field recall                       : per-type (name/national_id/birth_date/...)
                                             strict / normalized / fuzzy (Lev ≤ 2)
  - Length deltas                          : char count REF vs HYP
  - Worst-case / best-case examples per ablation
  - E2/E3/E4 vs E1 deltas (gain & regression cases)
"""

from __future__ import annotations

import csv
import pathlib
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

ASR_CSV = RESULTS_DIR / "phase3_breeze_asr_results.csv"
QUERIES_CSV = DATA_DIR / "phase3_queries.csv"

OUT_PER_QUERY = RESULTS_DIR / "phase3_asr_eval_per_query.csv"
OUT_SUMMARY = RESULTS_DIR / "phase3_asr_eval_summary.csv"
OUT_REPORT = RESULTS_DIR / "phase3_asr_eval_report.md"

ABLATIONS = ["e1", "e2"]
ABLATION_NAMES = {
    "e1": "E1 baseline",
    "e2": "E2 +保險詞",
}

LANGUAGES = ["mandarin", "hokkien", "codeswitch"]
SUBTYPES = ["A1", "A2", "B1", "B2", "C1", "C2", "D"]

# ============================================================
# Text normalisation
# ============================================================

PUNCT_PATTERN = re.compile(
    r"[\s，。、！？：；「」『』（）()【】《》<>“”‘’・…—\-,.!?:;\'\"]"
)
SPACE_PATTERN = re.compile(r"\s+")


def fullwidth_to_half(s: str) -> str:
    out = []
    for c in s:
        code = ord(c)
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            out.append(" ")
        else:
            out.append(c)
    return "".join(out)


def normalize_for_cer(s: str) -> str:
    s = fullwidth_to_half(s)
    s = PUNCT_PATTERN.sub("", s)
    return s.lower()


# ============================================================
# CN ↔ Arabic numeral normalisation (0–999, covers 民國年/月/日/手機/身分證以外場景)
# ============================================================

CN_BASE = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "兩": 2}


def cn_to_int(s: str) -> Optional[int]:
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s in CN_BASE:
        return CN_BASE[s]
    if s == "十":
        return 10
    if s.startswith("十") and len(s) == 2 and s[1] in CN_BASE:
        return 10 + CN_BASE[s[1]]
    if len(s) == 2 and s[1] == "十" and s[0] in CN_BASE:
        return CN_BASE[s[0]] * 10
    if len(s) == 3 and s[1] == "十" and s[0] in CN_BASE and s[2] in CN_BASE:
        return CN_BASE[s[0]] * 10 + CN_BASE[s[2]]
    # X百Y / X百Y十Z (rare in our set, best effort)
    if "百" in s:
        i = s.index("百")
        h = CN_BASE.get(s[:i])
        if h is None:
            return None
        rest = s[i + 1:]
        if not rest:
            return h * 100
        if rest in CN_BASE:
            return h * 100 + CN_BASE[rest]
        sub = cn_to_int(rest)
        return h * 100 + sub if sub is not None and sub < 100 else None
    return None


CN_DIGIT_RUN = r"[零一二三四五六七八九十百兩]+"
DATE_PATTERN = re.compile(
    rf"(民國|西元)?\s*({CN_DIGIT_RUN}|\d{{1,4}})\s*年\s*"
    rf"({CN_DIGIT_RUN}|\d{{1,2}})\s*月\s*"
    rf"({CN_DIGIT_RUN}|\d{{1,2}})\s*日?"
)
# 標準身分證：1英文字母 + 9數字
ID_PATTERN = re.compile(r"[A-Za-z][0-9]{9}")
# 保單號：英文 + 數字字串（保險）
POLICY_PATTERN = re.compile(r"[A-Za-z]{1,3}\d{6,12}")
# 手機：09xx-xxx-xxx 或 09xxxxxxxx
PHONE_PATTERN = re.compile(r"09\d{2}[\-\s]?\d{3}[\-\s]?\d{3}|09\d{8}")


def normalize_dates(s: str) -> str:
    def repl(m: re.Match) -> str:
        prefix = m.group(1) or ""
        y = cn_to_int(m.group(2))
        mo = cn_to_int(m.group(3))
        d = cn_to_int(m.group(4))
        if y is None or mo is None or d is None:
            return m.group(0)
        return f"{prefix}{y}年{mo}月{d}日"
    return DATE_PATTERN.sub(repl, s)


def normalize_for_pii(s: str) -> str:
    """Aggressive normalisation for PII matching."""
    s = fullwidth_to_half(s)
    s = normalize_dates(s)
    s = PUNCT_PATTERN.sub("", s)
    s = s.upper()  # 身分證大小寫不敏感
    return s


# ============================================================
# Edit distance / CER
# ============================================================

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref_n = normalize_for_cer(ref)
    hyp_n = normalize_for_cer(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return levenshtein(ref_n, hyp_n) / len(ref_n)


def cer_normalized(ref: str, hyp: str) -> float:
    ref_n = normalize_for_cer(normalize_dates(ref))
    hyp_n = normalize_for_cer(normalize_dates(hyp))
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return levenshtein(ref_n, hyp_n) / len(ref_n)


# ============================================================
# PII field extraction & matching
# ============================================================

def _substring_in(needle: str, hay: str) -> bool:
    return bool(needle) and needle in hay


def fuzzy_substring(needle: str, hay: str, max_dist: int) -> bool:
    """True if `needle` matches some hay substring with edit distance ≤ max_dist."""
    if not needle:
        return False
    if needle in hay:
        return True
    n, h = len(needle), len(hay)
    if h < n - max_dist:
        return False
    # Slide a window of length [n-max_dist, n+max_dist] and check Lev
    for win_len in range(max(1, n - max_dist), n + max_dist + 1):
        if win_len > h:
            break
        for i in range(h - win_len + 1):
            if levenshtein(needle, hay[i:i + win_len]) <= max_dist:
                return True
    return False


def match_pii(value: str, hyp: str, pii_type: str) -> dict:
    """Three match modes for one (value, hyp) pair."""
    v_norm = normalize_for_pii(value)
    h_norm = normalize_for_pii(hyp)

    res = {
        "strict": _substring_in(value, hyp),
        "normalized": _substring_in(v_norm, h_norm),
        "fuzzy": False,
    }

    # Pick fuzzy threshold by field type
    threshold = {
        "name": 1,
        "national_id": 1,
        "birth_date": 2,
        "policy_id": 1,
        "phone": 1,
        "address": 3,
    }.get(pii_type, 2)
    res["fuzzy"] = fuzzy_substring(v_norm, h_norm, threshold)
    return res


def parse_semicolon_field(s: str) -> list[str]:
    if not s or not s.strip():
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


# ============================================================
# Domain term recall
# ============================================================

def domain_term_match(term: str, hyp: str) -> dict:
    h_norm = normalize_for_pii(hyp)
    t_norm = normalize_for_pii(term)
    return {
        "strict": term in hyp,
        "normalized": t_norm in h_norm,
        "fuzzy": fuzzy_substring(t_norm, h_norm, 1),
    }


# ============================================================
# Load data
# ============================================================

def load_csv(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ============================================================
# Per-query evaluation
# ============================================================

@dataclass
class QueryEval:
    case_id: str
    language: str
    subtype: str
    dur_sec: float
    ref: str
    pii_types: list[str]
    pii_values: list[str]
    domain_terms: list[str]
    # Per-ablation metrics
    cer_strict: dict[str, float] = field(default_factory=dict)
    cer_norm: dict[str, float] = field(default_factory=dict)
    hyp_len: dict[str, int] = field(default_factory=dict)
    pii_match: dict[str, dict] = field(default_factory=dict)  # ablation -> {pii_type: {strict, norm, fuzzy}}
    domain_match: dict[str, dict] = field(default_factory=dict)


def evaluate_one(asr_row: dict, q_row: dict) -> QueryEval:
    ev = QueryEval(
        case_id=asr_row["case_id"],
        language=asr_row["language"],
        subtype=asr_row["subtype"],
        dur_sec=float(asr_row.get("dur_sec") or 0),
        ref=asr_row["ref"],
        pii_types=parse_semicolon_field(q_row.get("pii_types", "")),
        pii_values=parse_semicolon_field(q_row.get("pii_values", "")),
        domain_terms=parse_semicolon_field(q_row.get("domain_terms", "")),
    )
    ref = ev.ref

    for ab in ABLATIONS:
        hyp = asr_row.get(f"hyp_{ab}", "") or ""
        ev.cer_strict[ab] = cer(ref, hyp)
        ev.cer_norm[ab] = cer_normalized(ref, hyp)
        ev.hyp_len[ab] = len(normalize_for_cer(hyp))

        # PII per-type matches
        pii_results: dict[str, dict] = {}
        for ptype, pval in zip(ev.pii_types, ev.pii_values):
            pii_results[ptype] = match_pii(pval, hyp, ptype)
        ev.pii_match[ab] = pii_results

        # Domain term matches (one entry per term)
        ev.domain_match[ab] = {term: domain_term_match(term, hyp) for term in ev.domain_terms}

    return ev


# ============================================================
# Aggregation
# ============================================================

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(evals: list[QueryEval]) -> dict:
    """Return all aggregated tables as nested dicts."""
    agg: dict = {}

    # 1) Overall CER per ablation
    overall = {}
    for ab in ABLATIONS:
        overall[ab] = {
            "cer_strict": mean([e.cer_strict[ab] for e in evals]),
            "cer_norm": mean([e.cer_norm[ab] for e in evals]),
            "n": len(evals),
        }
    agg["overall"] = overall

    # 2) CER per ablation × language
    by_lang: dict = {}
    for lang in LANGUAGES:
        subset = [e for e in evals if e.language == lang]
        by_lang[lang] = {
            "n": len(subset),
            **{
                ab: {
                    "cer_strict": mean([e.cer_strict[ab] for e in subset]),
                    "cer_norm": mean([e.cer_norm[ab] for e in subset]),
                }
                for ab in ABLATIONS
            },
        }
    agg["by_language"] = by_lang

    # 3) CER per ablation × subtype
    by_sub: dict = {}
    for st in SUBTYPES:
        subset = [e for e in evals if e.subtype == st]
        by_sub[st] = {
            "n": len(subset),
            **{
                ab: {
                    "cer_strict": mean([e.cer_strict[ab] for e in subset]),
                    "cer_norm": mean([e.cer_norm[ab] for e in subset]),
                }
                for ab in ABLATIONS
            },
        }
    agg["by_subtype"] = by_sub

    # 4) PII recall per ablation × pii_type (3 match modes)
    pii_types_seen: set[str] = set()
    for e in evals:
        pii_types_seen.update(e.pii_types)

    by_pii: dict = {}
    for ptype in sorted(pii_types_seen):
        relevant = [e for e in evals if ptype in e.pii_types]
        n_total = len(relevant)
        ab_stats: dict = {"n_total": n_total}
        for ab in ABLATIONS:
            counts = {"strict": 0, "normalized": 0, "fuzzy": 0}
            for e in relevant:
                m = e.pii_match[ab].get(ptype, {})
                for mode in counts:
                    if m.get(mode):
                        counts[mode] += 1
            ab_stats[ab] = {
                **{f"{mode}_recall": counts[mode] / n_total if n_total else 0.0 for mode in counts},
                **{f"{mode}_count": counts[mode] for mode in counts},
            }
        by_pii[ptype] = ab_stats
    agg["by_pii"] = by_pii

    # 5) PII recall × ablation × language (only normalized mode for brevity)
    pii_by_lang: dict = {}
    for lang in LANGUAGES:
        subset = [e for e in evals if e.language == lang]
        for ptype in sorted(pii_types_seen):
            relevant = [e for e in subset if ptype in e.pii_types]
            if not relevant:
                continue
            row = {"language": lang, "pii_type": ptype, "n": len(relevant)}
            for ab in ABLATIONS:
                hits = sum(1 for e in relevant if e.pii_match[ab].get(ptype, {}).get("normalized"))
                row[f"{ab}_recall"] = hits / len(relevant)
            pii_by_lang[(lang, ptype)] = row
    agg["pii_by_lang"] = pii_by_lang

    # 6) Domain term recall per ablation
    all_terms = []
    for e in evals:
        for term in e.domain_terms:
            all_terms.append((e.case_id, term))
    n_term_instances = len(all_terms)
    domain_stats: dict = {"n_instances": n_term_instances, "n_unique_terms": len({t for _, t in all_terms})}
    for ab in ABLATIONS:
        counts = {"strict": 0, "normalized": 0, "fuzzy": 0}
        for e in evals:
            for term in e.domain_terms:
                m = e.domain_match[ab].get(term, {})
                for mode in counts:
                    if m.get(mode):
                        counts[mode] += 1
        domain_stats[ab] = {
            f"{mode}_recall": counts[mode] / n_term_instances if n_term_instances else 0.0
            for mode in counts
        } | {f"{mode}_count": counts[mode] for mode in counts}
    agg["domain_terms"] = domain_stats

    # 7) Per-term breakdown (which insurance terms benefit most from prompts?)
    per_term: dict = defaultdict(lambda: {"n": 0, **{ab: 0 for ab in ABLATIONS}})
    for e in evals:
        for term in e.domain_terms:
            per_term[term]["n"] += 1
            for ab in ABLATIONS:
                if e.domain_match[ab].get(term, {}).get("normalized"):
                    per_term[term][ab] += 1
    agg["per_term"] = dict(per_term)

    # 8) Best/worst per ablation
    worst: dict = {}
    for ab in ABLATIONS:
        ranked = sorted(evals, key=lambda e: -e.cer_norm[ab])
        worst[ab] = [(e.case_id, e.cer_norm[ab]) for e in ranked[:10]]
    agg["worst_cases"] = worst

    # 9) E2/E3/E4 vs E1 deltas per query
    diffs: dict = {ab: [] for ab in ABLATIONS if ab != "e1"}
    for ab in diffs:
        for e in evals:
            diffs[ab].append((e.case_id, e.cer_norm["e1"] - e.cer_norm[ab]))  # >0 = improvement
    # Top 5 gain & top 5 regression per ablation
    deltas: dict = {}
    for ab, lst in diffs.items():
        gains = sorted(lst, key=lambda x: -x[1])[:5]
        regressions = sorted(lst, key=lambda x: x[1])[:5]
        deltas[ab] = {"gain": gains, "regression": regressions}
    agg["deltas"] = deltas

    return agg


# ============================================================
# Output writers
# ============================================================

def fmt_pct(x: float) -> str:
    return f"{x * 100:5.2f}%"


def write_per_query_csv(evals: list[QueryEval], path: pathlib.Path) -> None:
    fieldnames = [
        "case_id", "language", "subtype", "dur_sec",
        "pii_types", "domain_terms",
        *[f"cer_strict_{ab}" for ab in ABLATIONS],
        *[f"cer_norm_{ab}" for ab in ABLATIONS],
        *[f"hyp_len_{ab}" for ab in ABLATIONS],
        # PII pass count per ablation (across all PII fields in this query, normalized mode)
        *[f"pii_pass_{ab}" for ab in ABLATIONS],
        *[f"pii_total_{ab}" for ab in ABLATIONS],
        *[f"domain_pass_{ab}" for ab in ABLATIONS],
        *[f"domain_total_{ab}" for ab in ABLATIONS],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for e in evals:
            row = {
                "case_id": e.case_id,
                "language": e.language,
                "subtype": e.subtype,
                "dur_sec": e.dur_sec,
                "pii_types": ";".join(e.pii_types),
                "domain_terms": ";".join(e.domain_terms),
            }
            for ab in ABLATIONS:
                row[f"cer_strict_{ab}"] = round(e.cer_strict[ab], 4)
                row[f"cer_norm_{ab}"] = round(e.cer_norm[ab], 4)
                row[f"hyp_len_{ab}"] = e.hyp_len[ab]
                pii_pass = sum(1 for m in e.pii_match[ab].values() if m.get("normalized"))
                row[f"pii_pass_{ab}"] = pii_pass
                row[f"pii_total_{ab}"] = len(e.pii_match[ab])
                d_pass = sum(1 for m in e.domain_match[ab].values() if m.get("normalized"))
                row[f"domain_pass_{ab}"] = d_pass
                row[f"domain_total_{ab}"] = len(e.domain_match[ab])
            w.writerow(row)


def write_summary_csv(agg: dict, path: pathlib.Path) -> None:
    rows: list[dict] = []
    # overall
    for ab, stats in agg["overall"].items():
        rows.append({"section": "overall", "key": "all", "ablation": ab,
                     "metric": "cer_strict", "value": round(stats["cer_strict"], 4), "n": stats["n"]})
        rows.append({"section": "overall", "key": "all", "ablation": ab,
                     "metric": "cer_norm", "value": round(stats["cer_norm"], 4), "n": stats["n"]})
    # by_language
    for lang, stats in agg["by_language"].items():
        for ab in ABLATIONS:
            rows.append({"section": "by_language", "key": lang, "ablation": ab,
                         "metric": "cer_strict", "value": round(stats[ab]["cer_strict"], 4), "n": stats["n"]})
            rows.append({"section": "by_language", "key": lang, "ablation": ab,
                         "metric": "cer_norm", "value": round(stats[ab]["cer_norm"], 4), "n": stats["n"]})
    # by_subtype
    for st, stats in agg["by_subtype"].items():
        for ab in ABLATIONS:
            rows.append({"section": "by_subtype", "key": st, "ablation": ab,
                         "metric": "cer_norm", "value": round(stats[ab]["cer_norm"], 4), "n": stats["n"]})
    # by_pii (normalized recall)
    for ptype, stats in agg["by_pii"].items():
        for ab in ABLATIONS:
            rows.append({"section": "by_pii", "key": ptype, "ablation": ab,
                         "metric": "normalized_recall",
                         "value": round(stats[ab]["normalized_recall"], 4),
                         "n": stats["n_total"]})
    # domain
    d = agg["domain_terms"]
    for ab in ABLATIONS:
        rows.append({"section": "domain", "key": "all", "ablation": ab,
                     "metric": "normalized_recall", "value": round(d[ab]["normalized_recall"], 4),
                     "n": d["n_instances"]})

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "key", "ablation", "metric", "value", "n"])
        w.writeheader()
        w.writerows(rows)


def write_report(agg: dict, evals: list[QueryEval], path: pathlib.Path) -> None:
    lines: list[str] = []

    def H(level: int, text: str) -> None:
        lines.append("#" * level + " " + text)
        lines.append("")

    def P(text: str = "") -> None:
        lines.append(text)
        lines.append("") if text else lines.append("")

    def TBL(headers: list[str], rows: list[list[str]]) -> None:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(x) for x in r) + " |")
        lines.append("")

    H(1, "Phase 3 Breeze-ASR-26 4-way Ablation Evaluation")
    P(f"**測試集**：{len(evals)} 條 query × 4 組 ablation = {len(evals) * 4} 次推論")
    P(f"**模型**：MediaTek-Research/Breeze-ASR-26 (Whisper-large-v2 fine-tune)")
    P(f"**環境**：Colab A100 / BF16 / num_beams=1 / chunk_length_s=30")
    P("")
    P("**4 組 ablation**：")
    for ab in ABLATIONS:
        P(f"- **{ab.upper()}** {ABLATION_NAMES[ab].split(' ', 1)[1]}")
    P("")
    P("**指標說明**：")
    P("- `CER` = 嚴格字元錯誤率（去標點/空白後計算 Levenshtein）")
    P("- `CER_norm` = 加上 中文數字↔阿拉伯數字 的日期標準化 後計算 (e.g. 民國七十年 ≡ 民國70年)")
    P("- `PII recall` = 該欄位個資在 hypothesis 中是否能被找到（normalized 模式：經過去標點/全形半形/中阿數字統一）")
    P("- `Domain term recall` = 保險術語在 hypothesis 中是否出現")
    P("---")

    # ---------- 1. Overall ----------
    H(2, "1. 總體 CER")
    rows = []
    for ab in ABLATIONS:
        s = agg["overall"][ab]
        rows.append([ABLATION_NAMES[ab], fmt_pct(s["cer_strict"]), fmt_pct(s["cer_norm"]), s["n"]])
    TBL(["Ablation", "CER (strict)", "CER (norm)", "N"], rows)
    best_ab = min(ABLATIONS, key=lambda a: agg["overall"][a]["cer_norm"])
    P(f"→ **最佳組合：{ABLATION_NAMES[best_ab]}**，CER_norm = "
      f"{fmt_pct(agg['overall'][best_ab]['cer_norm'])}")
    P("")

    # ---------- 2. By language ----------
    H(2, "2. 各語言條件下 CER (CER_norm)")
    rows = []
    for lang in LANGUAGES:
        s = agg["by_language"][lang]
        row = [lang, s["n"]] + [fmt_pct(s[ab]["cer_norm"]) for ab in ABLATIONS]
        rows.append(row)
    TBL(["language", "N"] + [ab.upper() for ab in ABLATIONS], rows)
    P("**觀察**：")
    for lang in LANGUAGES:
        s = agg["by_language"][lang]
        best = min(ABLATIONS, key=lambda a: s[a]["cer_norm"])
        worst = max(ABLATIONS, key=lambda a: s[a]["cer_norm"])
        P(f"- **{lang}** ({s['n']} 條)：最佳 {best.upper()}={fmt_pct(s[best]['cer_norm'])}, "
          f"最差 {worst.upper()}={fmt_pct(s[worst]['cer_norm'])}, "
          f"差距 {fmt_pct(s[worst]['cer_norm'] - s[best]['cer_norm'])}")
    P("")

    # ---------- 3. By subtype ----------
    H(2, "3. 各 subtype CER (CER_norm)")
    rows = []
    for st in SUBTYPES:
        s = agg["by_subtype"][st]
        if s["n"] == 0:
            continue
        rows.append([st, s["n"]] + [fmt_pct(s[ab]["cer_norm"]) for ab in ABLATIONS])
    TBL(["subtype", "N"] + [ab.upper() for ab in ABLATIONS], rows)
    P("**Subtype 對照**：A1=姓名+身分證+生日；A2=A1+保單號；B1=僅姓名；B2=姓名+保單號；")
    P("C1=姓名+地址；C2=姓名+電話；D=無 PII")
    P("")

    # ---------- 4. PII recall ----------
    H(2, "4. PII 各欄位召回率（normalized mode）")
    pii_order = ["name", "national_id", "birth_date", "policy_id", "phone", "address"]
    by_pii = agg["by_pii"]
    pii_rows = []
    for ptype in pii_order:
        if ptype not in by_pii:
            continue
        s = by_pii[ptype]
        row = [ptype, s["n_total"]]
        for ab in ABLATIONS:
            row.append(fmt_pct(s[ab]["normalized_recall"]))
        pii_rows.append(row)
    TBL(["PII 類型", "N (出現次數)"] + [ab.upper() for ab in ABLATIONS], pii_rows)
    P("")
    P("**3 種匹配模式對比**（headline 數字為 normalized）：")
    for ptype in pii_order:
        if ptype not in by_pii:
            continue
        s = by_pii[ptype]
        modes = ["strict", "normalized", "fuzzy"]
        rows = []
        for ab in ABLATIONS:
            rows.append([ABLATION_NAMES[ab]] +
                        [fmt_pct(s[ab][f"{m}_recall"]) for m in modes])
        H(3, f"4.{pii_order.index(ptype)+1} {ptype} (N={s['n_total']})")
        TBL(["Ablation", "strict", "normalized", "fuzzy"], rows)
    P("")

    # ---------- 5. PII × language ----------
    H(2, "5. PII recall × language（normalized）")
    pii_by_lang = agg["pii_by_lang"]
    rows = []
    for ptype in pii_order:
        for lang in LANGUAGES:
            r = pii_by_lang.get((lang, ptype))
            if not r:
                continue
            rows.append([ptype, lang, r["n"]] + [fmt_pct(r[f"{ab}_recall"]) for ab in ABLATIONS])
    TBL(["PII 類型", "language", "N"] + [ab.upper() for ab in ABLATIONS], rows)
    P("")

    # ---------- 6. Domain terms ----------
    H(2, "6. 保險術語召回率")
    d = agg["domain_terms"]
    rows = []
    for ab in ABLATIONS:
        rows.append([ABLATION_NAMES[ab],
                     fmt_pct(d[ab]["strict_recall"]),
                     fmt_pct(d[ab]["normalized_recall"]),
                     fmt_pct(d[ab]["fuzzy_recall"]),
                     d[ab]["normalized_count"]])
    TBL(["Ablation", "strict", "normalized", "fuzzy", "命中數"], rows)
    P(f"**N (術語實例) = {d['n_instances']}，獨立術語數 = {d['n_unique_terms']}**")
    P("")
    H(3, "6.1 各術語在不同 ablation 的命中數（normalized）")
    per_term = agg["per_term"]
    items = sorted(per_term.items(), key=lambda kv: -kv[1]["n"])
    rows = []
    for term, s in items:
        if s["n"] < 2:  # skip 一次性術語
            continue
        rows.append([term, s["n"]] + [s[ab] for ab in ABLATIONS])
    TBL(["術語", "出現次數"] + [ab.upper() for ab in ABLATIONS], rows)
    P("（僅顯示出現 ≥2 次的術語）")
    P("")

    # ---------- 7. Worst cases ----------
    H(2, "7. 最差 case（CER_norm 最高的 5 條，每組 ablation）")
    by_id = {e.case_id: e for e in evals}
    for ab in ABLATIONS:
        H(3, f"{ABLATION_NAMES[ab]}")
        rows = []
        for cid, val in agg["worst_cases"][ab][:5]:
            e = by_id[cid]
            rows.append([cid, e.language, e.subtype, fmt_pct(val),
                         (e.ref[:30] + "…") if len(e.ref) > 30 else e.ref])
        TBL(["case_id", "language", "subtype", "CER_norm", "REF (前 30 字)"], rows)

    # ---------- 8. Gains & regressions ----------
    others = [ab for ab in ABLATIONS if ab != "e1"]
    H(2, f"8. {'/'.join(a.upper() for a in others)} 相對 E1 的最大改善 / 退步")
    for ab in others:
        H(3, f"{ABLATION_NAMES[ab]} vs {ABLATION_NAMES['e1']}")
        d = agg["deltas"][ab]
        H(4, "Top 5 改善（E1 較差，此 ablation 救回）")
        rows = []
        for cid, delta in d["gain"]:
            if delta <= 0:
                continue
            e = by_id[cid]
            rows.append([cid, e.language, e.subtype,
                         fmt_pct(e.cer_norm["e1"]), fmt_pct(e.cer_norm[ab]),
                         fmt_pct(delta)])
        if rows:
            TBL(["case_id", "lang", "subtype", "E1 CER", f"{ab.upper()} CER", "改善幅度"], rows)
        else:
            P("_（無改善案例）_")

        H(4, "Top 5 退步（此 ablation 反而變差）")
        rows = []
        for cid, delta in d["regression"]:
            if delta >= 0:
                continue
            e = by_id[cid]
            rows.append([cid, e.language, e.subtype,
                         fmt_pct(e.cer_norm["e1"]), fmt_pct(e.cer_norm[ab]),
                         fmt_pct(-delta)])
        if rows:
            TBL(["case_id", "lang", "subtype", "E1 CER", f"{ab.upper()} CER", "退步幅度"], rows)
        else:
            P("_（無退步案例）_")

    # ---------- 9. Recommendations ----------
    H(2, "9. 結論與建議")
    overall = agg["overall"]
    best_ab = min(ABLATIONS, key=lambda a: overall[a]["cer_norm"])
    e1 = overall["e1"]["cer_norm"]
    bv = overall[best_ab]["cer_norm"]
    P(f"**最佳整體配置：{ABLATION_NAMES[best_ab]}**，"
      f"CER_norm 從 baseline 的 {fmt_pct(e1)} 降至 {fmt_pct(bv)}，"
      f"相對改善 {((e1 - bv) / e1 * 100 if e1 > 0 else 0):.1f}%。")
    P("")
    P("**Prompt 干預效果**：")
    for ab in [a for a in ABLATIONS if a != "e1"]:
        v = overall[ab]["cer_norm"]
        delta = e1 - v
        rel = (delta / e1 * 100) if e1 > 0 else 0
        P(f"- **{ABLATION_NAMES[ab]}** 相對 E1 絕對改善 {fmt_pct(delta)} (相對 {rel:.1f}%)")
    P("")

    # PII summary
    P("**PII 召回觀察**：")
    for ptype in pii_order:
        if ptype not in by_pii:
            continue
        s = by_pii[ptype]
        e1r = s["e1"]["normalized_recall"]
        bestabl = max(ABLATIONS, key=lambda a: s[a]["normalized_recall"])
        bestv = s[bestabl]["normalized_recall"]
        delta_str = "持平" if abs(bestv - e1r) < 0.005 else f"+{fmt_pct(bestv - e1r)}"
        P(f"- **{ptype}** (N={s['n_total']}): E1={fmt_pct(e1r)}, "
          f"最佳 {bestabl.upper()}={fmt_pct(bestv)} ({delta_str})")
    P("")

    P("---")
    P(f"_evaluation script: `benchmark/breeze_asr26/scripts/eval_asr_results.py`_")

    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> int:
    if not ASR_CSV.exists():
        print(f"ERROR: {ASR_CSV} not found", file=sys.stderr)
        return 2
    if not QUERIES_CSV.exists():
        print(f"ERROR: {QUERIES_CSV} not found", file=sys.stderr)
        return 2

    asr_rows = load_csv(ASR_CSV)
    q_rows = {r["case_id"]: r for r in load_csv(QUERIES_CSV)}

    print(f"Loaded {len(asr_rows)} ASR rows × {len(ABLATIONS)} ablations "
          f"({len(q_rows)} queries in queries CSV)")

    evals: list[QueryEval] = []
    for r in asr_rows:
        if r["case_id"] not in q_rows:
            print(f"WARN: {r['case_id']} not in queries CSV, skip")
            continue
        evals.append(evaluate_one(r, q_rows[r["case_id"]]))

    print(f"Evaluated {len(evals)} queries\n")

    agg = aggregate(evals)

    # Print headline to stdout
    print("=== Headline CER per ablation ===")
    for ab in ABLATIONS:
        s = agg["overall"][ab]
        print(f"  {ab.upper()} {ABLATION_NAMES[ab]:25s}  "
              f"CER_strict={fmt_pct(s['cer_strict'])}   CER_norm={fmt_pct(s['cer_norm'])}")
    print()
    print("=== PII normalized recall per type ===")
    for ptype, s in agg["by_pii"].items():
        line = f"  {ptype:12s} N={s['n_total']:3d}  "
        line += "  ".join(f"{ab.upper()}={fmt_pct(s[ab]['normalized_recall'])}" for ab in ABLATIONS)
        print(line)
    print()
    print("=== Domain term recall (normalized) ===")
    d = agg["domain_terms"]
    for ab in ABLATIONS:
        print(f"  {ab.upper()}: {fmt_pct(d[ab]['normalized_recall'])} "
              f"({d[ab]['normalized_count']}/{d['n_instances']})")
    print()

    write_per_query_csv(evals, OUT_PER_QUERY)
    write_summary_csv(agg, OUT_SUMMARY)
    write_report(agg, evals, OUT_REPORT)

    print(f"wrote {OUT_PER_QUERY.relative_to(ROOT.parent)}")
    print(f"wrote {OUT_SUMMARY.relative_to(ROOT.parent)}")
    print(f"wrote {OUT_REPORT.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
