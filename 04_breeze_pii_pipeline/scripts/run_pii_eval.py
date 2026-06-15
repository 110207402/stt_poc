"""
Phase 3 PII evaluation — M1 (OpenAI Privacy Filter) vs M2 (Azure GPT-4o-mini)
on E1 baseline + E2 +保險詞 hypothesis text.

Reads:
  results/phase3_breeze_asr_results.csv  (90 rows × hyp_e1, hyp_e2)
  data/phase3_queries.csv                (ground truth: pii_types, pii_values)

Writes:
  results/phase3_pii_detections.json     (raw detection cache, resumable)
  results/phase3_pii_eval_per_query.csv  (per-query metrics)
  results/phase3_pii_eval_summary.csv    (aggregated, long format)
  results/phase3_pii_eval_report.md      (human-readable analysis)

Methods:
  M1 = OpenAI Privacy Filter — HF: openai/privacy-filter
       8 native categories: private_person, private_address, private_email,
                             private_phone, private_url, private_date,
                             account_number, secret
       Mapped to our 6 GT types:
         name        ← private_person
         address     ← private_address
         phone       ← private_phone
         birth_date  ← private_date
         national_id ← account_number ∪ secret
         policy_id   ← account_number ∪ secret

  M2 = Azure GPT-4o-mini — Chat Completions w/ JSON schema
       Output 6 categories matching our GT directly:
         name, national_id, birth_date, policy_id, phone, address

Metrics:
  - Per-PII-type recall (per method × ablation)
  - End-to-end recall (ASR pipeline + PII method): did GT value get flagged?
  - False positive rate on D-subtype queries (no PII → any detection = FP)
  - Method comparison: M1 vs M2 agreement / disagreement
  - Upstream attribution: ASR-mangled vs PII-missed
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

# Reuse normalization + fuzzy match from ASR eval (same dir)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_asr_results import (  # noqa: E402
    normalize_for_pii,
    fuzzy_substring,
    levenshtein,
    parse_semicolon_field,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

ASR_CSV = RESULTS_DIR / "phase3_breeze_asr_results.csv"
QUERIES_CSV = DATA_DIR / "phase3_queries.csv"

CACHE_PATH = RESULTS_DIR / "phase3_pii_detections.json"
OUT_PER_QUERY = RESULTS_DIR / "phase3_pii_eval_per_query.csv"
OUT_SUMMARY = RESULTS_DIR / "phase3_pii_eval_summary.csv"
OUT_REPORT = RESULTS_DIR / "phase3_pii_eval_report.md"

ABLATIONS = ["e1", "e2"]
ABLATION_NAMES = {"e1": "E1 baseline", "e2": "E2 +保險詞"}
METHODS = ["m1", "m2"]
METHOD_NAMES = {
    "m1": "M1 OpenAI Privacy Filter",
    "m2": "M2 Azure GPT-4o-mini",
}

GT_TYPES = ["name", "national_id", "birth_date", "policy_id", "phone", "address"]

# M1 → GT type mapping
M1_TO_GT: dict[str, list[str]] = {
    "private_person": ["name"],
    "private_address": ["address"],
    "private_phone": ["phone"],
    "private_date": ["birth_date"],
    "account_number": ["national_id", "policy_id"],
    "secret": ["national_id", "policy_id"],
    # private_email / private_url not used
}

# Fuzzy thresholds (Lev distance)
MATCH_THRESHOLD: dict[str, int] = {
    "name": 1,
    "national_id": 1,
    "birth_date": 2,
    "policy_id": 2,
    "phone": 2,
    "address": 4,
}

# ============================================================
# M2 prompt + JSON schema
# ============================================================

M2_SYSTEM = """你是台灣金融客服文本的個資抽取助手。從輸入文字中找出所有個人資料，按 6 大類分類輸出 JSON。

類別定義：
- name: 客戶姓名（中文 2-4 字，例如 陳大天、林淑芬）
- national_id: 中華民國身分證字號（一個英文字母 + 9 位數字，例如 A123456789）
- birth_date: 出生年月日（民國或西元，例如 民國七十年三月五日 / 民國70年3月5日 / 1981/03/05）
- policy_id: 保險保單號碼（英文+數字+連字號組合，例如 KGI-2024-001234、KL-2020-987654、PA12345678）
- phone: 電話號碼（手機 09xx-xxx-xxx 或市話 02-xxxx-xxxx，注意 ASR 可能會漏前導 0）
- address: 地址（台灣縣市區/路段/巷/弄/號/樓，例如 台北市大安區忠孝東路四段100號5樓）

規則：
1. 只抽取真的出現在輸入文字中的字串（不要補完不存在的內容）。
2. 同一類別有多筆就全部列出。
3. 沒命中的類別給空陣列 []。
4. 不解釋，只輸出 JSON。"""

M2_SCHEMA = {
    "name": "kgi_pii_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name":        {"type": "array", "items": {"type": "string"}},
            "national_id": {"type": "array", "items": {"type": "string"}},
            "birth_date":  {"type": "array", "items": {"type": "string"}},
            "policy_id":   {"type": "array", "items": {"type": "string"}},
            "phone":       {"type": "array", "items": {"type": "string"}},
            "address":     {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "national_id", "birth_date", "policy_id", "phone", "address"],
        "additionalProperties": False,
    },
}

# ============================================================
# M1: OpenAI Privacy Filter (lazy load via ONNX runtime)
# ============================================================
# The released `transformers` versions don't recognize the
# `openai_privacy_filter` architecture, and the repo ships no remote
# code. We load the published ONNX export + tokenizer.json directly
# and decode BIOES tags ourselves. Keeps the rest of the project
# pinned to transformers 4.57.x.

_m1_state: dict | None = None  # {"sess": ort.InferenceSession, "tok": Tokenizer, "id2label": dict[int,str]}


def m1_load():
    global _m1_state
    if _m1_state is not None:
        return _m1_state
    import json as _json
    import os as _os
    import numpy as _np  # noqa: F401  (imported here so failure shows up at load time)
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    print("[m1] loading openai/privacy-filter (ONNX) ...", flush=True)
    local = snapshot_download(
        "openai/privacy-filter",
        allow_patterns=[
            "onnx/model_quantized.onnx",
            "onnx/model_quantized.onnx_data",
            "tokenizer*",
            "config.json",
        ],
    )
    tok = Tokenizer.from_file(_os.path.join(local, "tokenizer.json"))
    sess = ort.InferenceSession(
        _os.path.join(local, "onnx/model_quantized.onnx"),
        providers=["CPUExecutionProvider"],
    )
    cfg = _json.load(open(_os.path.join(local, "config.json")))
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    _m1_state = {"sess": sess, "tok": tok, "id2label": id2label}
    print("[m1] ready", flush=True)
    return _m1_state


def _strip_bioes(label: str) -> tuple[str, str]:
    """('B-private_person') -> ('B', 'private_person'); ('O') -> ('O', '')."""
    if label == "O" or "-" not in label:
        return ("O" if label == "O" else label, "")
    prefix, _, cat = label.partition("-")
    return prefix, cat


def _decode_bioes_spans(labels: list[str], offsets: list[tuple[int, int]],
                        scores: list[float], text: str) -> list[dict]:
    """Walk the tag sequence and emit one entity per coherent span.

    Boundaries are not strictly checked (we accept lax B*/I*/E*/S* — the
    constrained Viterbi in the official model would clean these up, but
    argmax with a lenient walker is good enough for recall-oriented eval).
    """
    spans: list[dict] = []
    cur_cat: str | None = None
    cur_start: int | None = None
    cur_end: int | None = None
    cur_scores: list[float] = []

    def _flush():
        nonlocal cur_cat, cur_start, cur_end, cur_scores
        if cur_cat is not None and cur_start is not None and cur_end is not None and cur_end > cur_start:
            spans.append({
                "category": cur_cat,
                "text": text[cur_start:cur_end],
                "start": int(cur_start),
                "end": int(cur_end),
                "score": float(sum(cur_scores) / max(1, len(cur_scores))),
            })
        cur_cat = None; cur_start = None; cur_end = None; cur_scores = []

    for lab, (s, e), sc in zip(labels, offsets, scores):
        prefix, cat = _strip_bioes(lab)
        if prefix == "O" or cat == "":
            _flush()
            continue
        if s == 0 and e == 0:  # special tokens — skip
            continue
        if prefix == "S":
            _flush()
            cur_cat, cur_start, cur_end, cur_scores = cat, s, e, [sc]
            _flush()
        elif prefix == "B":
            _flush()
            cur_cat, cur_start, cur_end, cur_scores = cat, s, e, [sc]
        elif prefix == "I":
            if cur_cat == cat:
                cur_end = e
                cur_scores.append(sc)
            else:
                # treat stray I-X as a span start (lenient)
                _flush()
                cur_cat, cur_start, cur_end, cur_scores = cat, s, e, [sc]
        elif prefix == "E":
            if cur_cat == cat:
                cur_end = e
                cur_scores.append(sc)
                _flush()
            else:
                _flush()
                cur_cat, cur_start, cur_end, cur_scores = cat, s, e, [sc]
                _flush()
        else:
            _flush()

    _flush()
    return spans


def m1_detect(text: str) -> list[dict]:
    import numpy as np
    state = m1_load()
    sess = state["sess"]; tok = state["tok"]; id2label = state["id2label"]

    enc = tok.encode(text)
    ids = np.array([enc.ids], dtype="int64")
    mask = np.ones_like(ids)
    logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]  # [T, 33]
    # Softmax for confidence scoring
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    pred = probs.argmax(axis=-1)
    labels = [id2label[int(p)] for p in pred]
    scores = [float(probs[i, int(pred[i])]) for i in range(len(pred))]
    offsets = [tuple(o) for o in enc.offsets]
    return _decode_bioes_spans(labels, offsets, scores, text)


# ============================================================
# M2: Azure GPT-4o-mini (lazy load)
# ============================================================

_m2_client = None


def m2_load():
    global _m2_client
    if _m2_client is not None:
        return _m2_client
    from azure_client import get_chat_client
    print("[m2] loading Azure chat client ...", flush=True)
    _m2_client = get_chat_client()
    print("[m2] ready", flush=True)
    return _m2_client


def m2_detect(text: str, max_retries: int = 6) -> dict:
    from azure_client import CHAT_DEPLOYMENT
    client = m2_load()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": M2_SYSTEM},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_schema", "json_schema": M2_SCHEMA},
                temperature=0,
                timeout=60,
            )
            content = resp.choices[0].message.content or "{}"
            parsed = json.loads(content)
            # ensure all keys present
            for k in GT_TYPES:
                parsed.setdefault(k, [])
                if not isinstance(parsed[k], list):
                    parsed[k] = []
            parsed["_usage"] = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            return parsed
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            etype = type(e).__name__.lower()
            retryable = any(s in msg or s in etype for s in
                            ["rate", "429", "timeout", "timed out", "connection", "503"])
            if retryable and attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                print(f"  [m2] retry {attempt+1}/{max_retries} in {wait}s "
                      f"({type(e).__name__})", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"M2 failed after {max_retries} retries: {last_err}")


# ============================================================
# Detection cache
# ============================================================

def cache_load() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            print(f"  [cache] corrupted, starting fresh", flush=True)
    return {}


def cache_save(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_key(case_id: str, ablation: str, method: str) -> str:
    return f"{case_id}:{ablation}:{method}"


# ============================================================
# Detection runner
# ============================================================

def run_detections(asr_rows: list[dict], methods: list[str], force: bool) -> dict:
    cache = cache_load()
    n_calls = 0
    n_skip = 0

    for i, r in enumerate(asr_rows, 1):
        cid = r["case_id"]
        for ab in ABLATIONS:
            hyp = (r.get(f"hyp_{ab}", "") or "").strip()
            for m in methods:
                k = cache_key(cid, ab, m)
                if not force and k in cache:
                    n_skip += 1
                    continue
                try:
                    if m == "m1":
                        det = m1_detect(hyp)
                    elif m == "m2":
                        det = m2_detect(hyp)
                    else:
                        raise ValueError(f"unknown method {m}")
                    cache[k] = det
                    n_calls += 1
                    if n_calls % 10 == 0:
                        cache_save(cache)
                        print(f"  [{i:03d}/{len(asr_rows)}] {cid} {ab} {m}: cached "
                              f"({n_calls} calls, {n_skip} skipped)", flush=True)
                except Exception as e:
                    print(f"  [{i:03d}/{len(asr_rows)}] {cid} {ab} {m} FAIL: {e}", flush=True)
                    cache[k] = {"_error": str(e)}
                    cache_save(cache)
        if i % 5 == 0:
            cache_save(cache)

    cache_save(cache)
    print(f"\nDetection done. {n_calls} new calls, {n_skip} cache hits.", flush=True)
    return cache


# ============================================================
# Unified detection format
# ============================================================

def m1_to_unified(det: list[dict]) -> list[dict]:
    """Map M1's 8-category output to our 6 GT types (one entity may map to multiple)."""
    out = []
    for ent in det:
        cat = ent.get("category", "")
        for gt in M1_TO_GT.get(cat, []):
            out.append({"category": gt, "text": ent.get("text", ""),
                        "raw_category": cat, "score": ent.get("score", 0)})
    return out


def m2_to_unified(det: dict) -> list[dict]:
    """Map M2's dict-of-lists to flat unified list."""
    out = []
    for cat in GT_TYPES:
        for txt in det.get(cat, []) or []:
            if isinstance(txt, str) and txt.strip():
                out.append({"category": cat, "text": txt.strip()})
    return out


def to_unified(det, method: str) -> list[dict]:
    if det is None:
        return []
    if isinstance(det, dict) and det.get("_error"):
        return []
    if method == "m1":
        return m1_to_unified(det)
    if method == "m2":
        return m2_to_unified(det if isinstance(det, dict) else {})
    return []


# ============================================================
# Matching: GT vs detected
# ============================================================

def is_match(gt_value: str, det_text: str, pii_type: str) -> str:
    """Return match level: 'exact' | 'normalized' | 'fuzzy' | 'no'."""
    if not gt_value or not det_text:
        return "no"
    if gt_value in det_text or det_text in gt_value:
        return "exact"
    v_norm = normalize_for_pii(gt_value)
    d_norm = normalize_for_pii(det_text)
    if v_norm in d_norm or d_norm in v_norm:
        return "normalized"
    threshold = MATCH_THRESHOLD.get(pii_type, 2)
    # bidirectional fuzzy substring
    if fuzzy_substring(v_norm, d_norm, threshold) or fuzzy_substring(d_norm, v_norm, threshold):
        return "fuzzy"
    return "no"


def evaluate_query_method(gt_pii: list[tuple[str, str]],
                          unified: list[dict],
                          subtype: str) -> dict:
    """Compute recall/precision per PII type for one (case, ablation, method) tuple.

    Returns:
      {
        'recall':    {pii_type: 1/0 or None if not in GT},
        'match_level': {pii_type: 'exact'|'normalized'|'fuzzy'|'no'},
        'fp_count':  {pii_type: int},   # detected entities of this type that don't match any GT
        'tp_count':  {pii_type: int},   # detected entities matching some GT
        'det_total': {pii_type: int},   # total detections of this type
      }
    """
    recall: dict = {}
    level: dict = {}
    matched_dets: set[int] = set()  # indexes in `unified` already credited

    # For each GT (type, value), find best detected match
    gt_by_type: dict[str, list[str]] = defaultdict(list)
    for t, v in gt_pii:
        gt_by_type[t].append(v)

    for ptype, values in gt_by_type.items():
        any_hit = False
        best_level = "no"
        for v in values:
            for i, d in enumerate(unified):
                if d["category"] != ptype:
                    continue
                lv = is_match(v, d["text"], ptype)
                if lv != "no":
                    any_hit = True
                    matched_dets.add(i)
                    # rank: exact > normalized > fuzzy
                    rank = {"exact": 3, "normalized": 2, "fuzzy": 1, "no": 0}
                    if rank[lv] > rank[best_level]:
                        best_level = lv
        recall[ptype] = 1 if any_hit else 0
        level[ptype] = best_level

    # FP / TP per type
    fp_count: Counter = Counter()
    tp_count: Counter = Counter()
    det_total: Counter = Counter()
    for i, d in enumerate(unified):
        det_total[d["category"]] += 1
        if i in matched_dets:
            tp_count[d["category"]] += 1
        else:
            fp_count[d["category"]] += 1

    return {
        "recall": recall,
        "match_level": level,
        "fp_count": dict(fp_count),
        "tp_count": dict(tp_count),
        "det_total": dict(det_total),
        "n_detections": len(unified),
    }


# ============================================================
# Aggregation
# ============================================================

def aggregate(per_query: list[dict]) -> dict:
    """per_query is list of dicts with case_id, ablation, method, language, subtype + metrics."""
    agg: dict = {}

    # 1) Recall per (method, ablation, pii_type)
    recall_table: dict = {}
    for m in METHODS:
        for ab in ABLATIONS:
            subset = [r for r in per_query if r["method"] == m and r["ablation"] == ab]
            for pt in GT_TYPES:
                relevant = [r for r in subset if pt in r["recall"]]
                hits = sum(1 for r in relevant if r["recall"][pt] == 1)
                recall_table[(m, ab, pt)] = {
                    "n": len(relevant),
                    "hits": hits,
                    "recall": hits / len(relevant) if relevant else 0.0,
                }
    agg["recall_table"] = recall_table

    # 2) Recall per (method, ablation, language, pii_type)
    recall_by_lang: dict = {}
    langs = sorted(set(r["language"] for r in per_query))
    for m in METHODS:
        for ab in ABLATIONS:
            for lang in langs:
                subset = [r for r in per_query
                          if r["method"] == m and r["ablation"] == ab and r["language"] == lang]
                for pt in GT_TYPES:
                    relevant = [r for r in subset if pt in r["recall"]]
                    if not relevant:
                        continue
                    hits = sum(1 for r in relevant if r["recall"][pt] == 1)
                    recall_by_lang[(m, ab, lang, pt)] = {
                        "n": len(relevant),
                        "hits": hits,
                        "recall": hits / len(relevant),
                    }
    agg["recall_by_lang"] = recall_by_lang

    # 3) False positives on D queries (no PII → all detections are FP)
    fp_d: dict = {}
    for m in METHODS:
        for ab in ABLATIONS:
            subset = [r for r in per_query
                      if r["method"] == m and r["ablation"] == ab and r["subtype"] == "D"]
            n_d = len(subset)
            total_dets = sum(r["n_detections"] for r in subset)
            cases_with_fp = sum(1 for r in subset if r["n_detections"] > 0)
            per_type: Counter = Counter()
            for r in subset:
                for pt, n in r["det_total"].items():
                    per_type[pt] += n
            fp_d[(m, ab)] = {
                "n_d_queries": n_d,
                "total_fp_detections": total_dets,
                "cases_with_fp": cases_with_fp,
                "fp_rate_per_query": total_dets / n_d if n_d else 0.0,
                "case_fp_rate": cases_with_fp / n_d if n_d else 0.0,
                "per_type": dict(per_type),
            }
    agg["fp_d"] = fp_d

    # 4) Method × ablation overall recall (micro-average across all PII instances)
    overall: dict = {}
    for m in METHODS:
        for ab in ABLATIONS:
            subset = [r for r in per_query if r["method"] == m and r["ablation"] == ab]
            total_gt = 0
            total_hits = 0
            for r in subset:
                for pt, val in r["recall"].items():
                    total_gt += 1
                    total_hits += val
            overall[(m, ab)] = {
                "n_pii_instances": total_gt,
                "hits": total_hits,
                "recall": total_hits / total_gt if total_gt else 0.0,
            }
    agg["overall"] = overall

    # 5) Match level distribution (exact / normalized / fuzzy / no)
    levels: dict = {}
    for m in METHODS:
        for ab in ABLATIONS:
            subset = [r for r in per_query if r["method"] == m and r["ablation"] == ab]
            cnt: Counter = Counter()
            for r in subset:
                for pt, lv in r["match_level"].items():
                    cnt[lv] += 1
            levels[(m, ab)] = dict(cnt)
    agg["match_levels"] = levels

    # 6) Method agreement per query (for each (case, ablation), do M1 and M2 agree?)
    agreement: list = []
    for cid, ab in {(r["case_id"], r["ablation"]) for r in per_query}:
        m1 = next((r for r in per_query if r["case_id"] == cid and r["ablation"] == ab and r["method"] == "m1"), None)
        m2 = next((r for r in per_query if r["case_id"] == cid and r["ablation"] == ab and r["method"] == "m2"), None)
        if not m1 or not m2:
            continue
        for pt in GT_TYPES:
            r1 = m1["recall"].get(pt)
            r2 = m2["recall"].get(pt)
            if r1 is None and r2 is None:
                continue
            agreement.append({
                "case_id": cid, "ablation": ab, "pii_type": pt,
                "m1": r1, "m2": r2,
                "agree": (r1 == r2),
            })
    agg["agreement"] = agreement

    return agg


# ============================================================
# Output writers
# ============================================================

def fmt_pct(x: float) -> str:
    return f"{x * 100:5.2f}%"


def write_per_query_csv(per_query: list[dict], path: pathlib.Path) -> None:
    fieldnames = [
        "case_id", "ablation", "method", "language", "subtype",
        "n_detections",
        *[f"recall_{pt}" for pt in GT_TYPES],
        *[f"level_{pt}" for pt in GT_TYPES],
        *[f"det_{pt}" for pt in GT_TYPES],
        *[f"fp_{pt}" for pt in GT_TYPES],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_query:
            row = {
                "case_id": r["case_id"],
                "ablation": r["ablation"],
                "method": r["method"],
                "language": r["language"],
                "subtype": r["subtype"],
                "n_detections": r["n_detections"],
            }
            for pt in GT_TYPES:
                row[f"recall_{pt}"] = r["recall"].get(pt, "")
                row[f"level_{pt}"] = r["match_level"].get(pt, "")
                row[f"det_{pt}"] = r["det_total"].get(pt, 0)
                row[f"fp_{pt}"] = r["fp_count"].get(pt, 0)
            w.writerow(row)


def write_summary_csv(agg: dict, path: pathlib.Path) -> None:
    rows: list[dict] = []
    for (m, ab, pt), s in agg["recall_table"].items():
        rows.append({
            "section": "recall_by_type", "method": m, "ablation": ab,
            "pii_type": pt, "language": "", "metric": "recall",
            "value": round(s["recall"], 4), "n": s["n"], "hits": s["hits"],
        })
    for (m, ab, lang, pt), s in agg["recall_by_lang"].items():
        rows.append({
            "section": "recall_by_lang", "method": m, "ablation": ab,
            "pii_type": pt, "language": lang, "metric": "recall",
            "value": round(s["recall"], 4), "n": s["n"], "hits": s["hits"],
        })
    for (m, ab), s in agg["overall"].items():
        rows.append({
            "section": "overall", "method": m, "ablation": ab,
            "pii_type": "", "language": "", "metric": "micro_recall",
            "value": round(s["recall"], 4), "n": s["n_pii_instances"], "hits": s["hits"],
        })
    for (m, ab), s in agg["fp_d"].items():
        rows.append({
            "section": "fp_d", "method": m, "ablation": ab,
            "pii_type": "", "language": "", "metric": "fp_per_query",
            "value": round(s["fp_rate_per_query"], 4), "n": s["n_d_queries"],
            "hits": s["total_fp_detections"],
        })
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "method", "ablation", "pii_type",
                                          "language", "metric", "value", "n", "hits"])
        w.writeheader()
        w.writerows(rows)


def write_report(agg: dict, per_query: list[dict], path: pathlib.Path) -> None:
    lines: list[str] = []

    def H(level: int, text: str) -> None:
        lines.append("#" * level + " " + text); lines.append("")

    def P(text: str = "") -> None:
        lines.append(text); lines.append("")

    def TBL(headers: list[str], rows: list[list]) -> None:
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(x) for x in r) + " |")
        lines.append("")

    H(1, "Phase 3 PII Detection Evaluation — M1 vs M2")
    P(f"**測試集**：{len({(r['case_id'], r['ablation']) for r in per_query})} (case × ablation) "
      f"× {len(METHODS)} methods = {len(per_query)} 次偵測")
    P(f"**ASR 上游**：Breeze-ASR-26 在 E1 baseline + E2 +保險詞 兩組設定的輸出")
    P("**PII 方法**：")
    P("- **M1** OpenAI Privacy Filter (HF: `openai/privacy-filter`) — 8 native categories 映射成我們 6 類")
    P("- **M2** Azure GPT-4o-mini — JSON schema 強制輸出 6 類")
    P("**比對標準**（容許 ASR 帶來的字元錯誤）：")
    P("- **exact**：偵測到的字串完全包含 GT 值")
    P("- **normalized**：去標點/全形半形/中阿數字統一後包含")
    P("- **fuzzy**：normalized 後 Levenshtein ≤ 類別 threshold (name=1, id=1, date=2, addr=4 ...)")
    P("- **no**：沒命中")
    P("---")

    # ============== 1. Overall recall ==============
    H(2, "1. 整體 micro-recall（所有 PII instance 池在一起）")
    rows = []
    for m in METHODS:
        for ab in ABLATIONS:
            s = agg["overall"][(m, ab)]
            rows.append([METHOD_NAMES[m], ABLATION_NAMES[ab],
                         s["n_pii_instances"], s["hits"], fmt_pct(s["recall"])])
    TBL(["Method", "Ablation", "PII 總數", "命中", "Recall"], rows)

    # ============== 2. Recall by PII type ==============
    H(2, "2. 各 PII 類別 recall（method × ablation）")
    headers = ["PII 類型", "N"]
    for m in METHODS:
        for ab in ABLATIONS:
            headers.append(f"{m.upper()}/{ab.upper()}")
    rows = []
    for pt in GT_TYPES:
        n = agg["recall_table"][(METHODS[0], ABLATIONS[0], pt)]["n"]
        if n == 0:
            continue
        row = [pt, n]
        for m in METHODS:
            for ab in ABLATIONS:
                s = agg["recall_table"][(m, ab, pt)]
                row.append(fmt_pct(s["recall"]))
        rows.append(row)
    TBL(headers, rows)

    # ============== 3. M1 vs M2 head-to-head ==============
    H(2, "3. M1 vs M2 同 ablation 直接對比")
    for ab in ABLATIONS:
        H(3, f"{ABLATION_NAMES[ab]}")
        rows = []
        for pt in GT_TYPES:
            n = agg["recall_table"][("m1", ab, pt)]["n"]
            if n == 0:
                continue
            r1 = agg["recall_table"][("m1", ab, pt)]["recall"]
            r2 = agg["recall_table"][("m2", ab, pt)]["recall"]
            delta = r2 - r1
            rows.append([pt, n, fmt_pct(r1), fmt_pct(r2),
                         f"+{fmt_pct(delta)}" if delta > 0
                         else (fmt_pct(delta) if delta < 0 else "持平")])
        TBL(["PII 類型", "N", "M1 recall", "M2 recall", "M2-M1 差距"], rows)

    # ============== 4. Recall by language ==============
    H(2, "4. Recall × language（normalized / micro per language-PII pair）")
    for ab in ABLATIONS:
        H(3, f"{ABLATION_NAMES[ab]}")
        langs = sorted({key[2] for key in agg["recall_by_lang"].keys()})
        rows = []
        for pt in GT_TYPES:
            for lang in langs:
                key = ("m1", ab, lang, pt)
                if key not in agg["recall_by_lang"]:
                    continue
                s1 = agg["recall_by_lang"][key]
                s2 = agg["recall_by_lang"][("m2", ab, lang, pt)]
                rows.append([pt, lang, s1["n"], fmt_pct(s1["recall"]), fmt_pct(s2["recall"])])
        TBL(["PII 類型", "language", "N", "M1", "M2"], rows)

    # ============== 5. False positives on D queries ==============
    H(2, "5. 假陽性分析（D-subtype 30 條無 PII query）")
    rows = []
    for m in METHODS:
        for ab in ABLATIONS:
            s = agg["fp_d"][(m, ab)]
            rows.append([METHOD_NAMES[m], ABLATION_NAMES[ab],
                         s["n_d_queries"], s["total_fp_detections"],
                         s["cases_with_fp"], fmt_pct(s["case_fp_rate"]),
                         f"{s['fp_rate_per_query']:.2f}"])
    TBL(["Method", "Ablation", "N (D-cases)", "總 FP 數", "有 FP 的 case 數",
         "Case FP rate", "平均 FP/case"], rows)
    P("**理解**：D-subtype 全部沒有 PII，任何偵測都是假陽性。")
    P("`Case FP rate` = 至少有 1 筆假陽性的 case 比例；`平均 FP/case` = 每條 D query 平均偵測幾個假陽。")
    P("")
    P("**FP 類型分布**：")
    for m in METHODS:
        for ab in ABLATIONS:
            s = agg["fp_d"][(m, ab)]
            if not s["per_type"]:
                continue
            dist = ", ".join(f"{pt}={n}" for pt, n in
                             sorted(s["per_type"].items(), key=lambda kv: -kv[1]))
            P(f"- {METHOD_NAMES[m]} / {ABLATION_NAMES[ab]}: {dist}")
    P("")

    # ============== 6. Match level distribution ==============
    H(2, "6. 命中模式分布")
    P("（在所有有 PII 的 case 中，每個欄位的 recall 落在哪個 match level）")
    headers = ["Method", "Ablation", "exact", "normalized", "fuzzy", "no (miss)"]
    rows = []
    for m in METHODS:
        for ab in ABLATIONS:
            lv = agg["match_levels"].get((m, ab), {})
            tot = sum(lv.values()) or 1
            rows.append([METHOD_NAMES[m], ABLATION_NAMES[ab],
                         f"{lv.get('exact', 0)} ({fmt_pct(lv.get('exact', 0)/tot)})",
                         f"{lv.get('normalized', 0)} ({fmt_pct(lv.get('normalized', 0)/tot)})",
                         f"{lv.get('fuzzy', 0)} ({fmt_pct(lv.get('fuzzy', 0)/tot)})",
                         f"{lv.get('no', 0)} ({fmt_pct(lv.get('no', 0)/tot)})"])
    TBL(headers, rows)

    # ============== 7. Method agreement ==============
    H(2, "7. M1 / M2 一致性")
    agreement = agg["agreement"]
    by_ab: dict = defaultdict(list)
    for r in agreement:
        by_ab[r["ablation"]].append(r)
    rows = []
    for ab, lst in by_ab.items():
        n = len(lst)
        agree = sum(1 for r in lst if r["agree"])
        m1_only = sum(1 for r in lst if r["m1"] == 1 and r["m2"] == 0)
        m2_only = sum(1 for r in lst if r["m1"] == 0 and r["m2"] == 1)
        both = sum(1 for r in lst if r["m1"] == 1 and r["m2"] == 1)
        neither = sum(1 for r in lst if r["m1"] == 0 and r["m2"] == 0)
        rows.append([ABLATION_NAMES[ab], n,
                     fmt_pct(agree / n) if n else "—",
                     both, neither, m1_only, m2_only])
    TBL(["Ablation", "N (PII 欄位數)", "agreement",
         "兩者皆中", "兩者皆漏", "僅 M1 中", "僅 M2 中"], rows)

    # ============== 8. Conclusions ==============
    H(2, "8. 結論")
    best = None
    best_recall = -1
    for (m, ab), s in agg["overall"].items():
        if s["recall"] > best_recall:
            best_recall = s["recall"]; best = (m, ab)
    P(f"**最佳組合**：{METHOD_NAMES[best[0]]} 在 {ABLATION_NAMES[best[1]]} 上 micro-recall = {fmt_pct(best_recall)}")
    P("")
    # Recall delta E1 → E2 per method
    for m in METHODS:
        e1 = agg["overall"][(m, "e1")]["recall"]
        e2 = agg["overall"][(m, "e2")]["recall"]
        delta = e2 - e1
        verdict = ("E2 prompt 提升 PII recall" if delta > 0.01
                   else ("E2 prompt 反而拖累 PII recall" if delta < -0.01
                         else "E2 prompt 對 PII recall 無顯著影響"))
        P(f"- **{METHOD_NAMES[m]}**：E1={fmt_pct(e1)}, E2={fmt_pct(e2)}, "
          f"差距={fmt_pct(delta)} → {verdict}")
    P("")
    P("**FP 觀察**：")
    for m in METHODS:
        e1fp = agg["fp_d"][(m, "e1")]["case_fp_rate"]
        e2fp = agg["fp_d"][(m, "e2")]["case_fp_rate"]
        P(f"- **{METHOD_NAMES[m]}**：D queries case FP rate E1={fmt_pct(e1fp)} / E2={fmt_pct(e2fp)}")
    P("")
    P(f"_evaluation script: `benchmark/breeze_asr26/scripts/run_pii_eval.py`_")
    P(f"_raw cache: `{CACHE_PATH.relative_to(ROOT.parent)}`_")

    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="m1,m2",
                        help="comma-separated methods to run (default m1,m2)")
    parser.add_argument("--limit", type=int, default=0,
                        help="only first N rows (for smoke test)")
    parser.add_argument("--only", default="",
                        help="comma-separated case_ids to run only")
    parser.add_argument("--force", action="store_true",
                        help="re-run even if cached")
    parser.add_argument("--no-detect", action="store_true",
                        help="skip detection (use existing cache only) and just re-aggregate")
    args = parser.parse_args()

    if not ASR_CSV.exists():
        print(f"ERROR: {ASR_CSV} not found", file=sys.stderr); return 2
    if not QUERIES_CSV.exists():
        print(f"ERROR: {QUERIES_CSV} not found", file=sys.stderr); return 2

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    invalid = [m for m in methods if m not in METHODS]
    if invalid:
        print(f"ERROR: unknown method(s) {invalid}", file=sys.stderr); return 2

    asr_rows = list(csv.DictReader(ASR_CSV.open(encoding="utf-8")))
    q_rows = {r["case_id"]: r for r in csv.DictReader(QUERIES_CSV.open(encoding="utf-8"))}

    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        asr_rows = [r for r in asr_rows if r["case_id"] in wanted]
    elif args.limit > 0:
        asr_rows = asr_rows[: args.limit]

    print(f"Loaded {len(asr_rows)} ASR rows × {len(ABLATIONS)} ablations × "
          f"{len(methods)} methods = {len(asr_rows) * len(ABLATIONS) * len(methods)} detections")
    print(f"Methods: {methods}")
    print(f"Cache: {CACHE_PATH.relative_to(ROOT.parent)}\n")

    # ===== Run detections =====
    if args.no_detect:
        cache = cache_load()
        print(f"--no-detect: using existing cache ({len(cache)} entries)\n")
    else:
        cache = run_detections(asr_rows, methods, args.force)

    # ===== Build per_query metrics =====
    per_query: list[dict] = []
    for r in asr_rows:
        cid = r["case_id"]
        q = q_rows.get(cid)
        if not q:
            print(f"WARN: {cid} not in queries CSV, skip"); continue
        types = parse_semicolon_field(q.get("pii_types", ""))
        values = parse_semicolon_field(q.get("pii_values", ""))
        gt_pii = list(zip(types, values))
        for ab in ABLATIONS:
            for m in methods:
                k = cache_key(cid, ab, m)
                det = cache.get(k)
                if det is None:
                    print(f"WARN: no cache for {k}, skipping"); continue
                if isinstance(det, dict) and det.get("_error"):
                    print(f"WARN: error in {k}: {det['_error']}, skipping"); continue
                unified = to_unified(det, m)
                metrics = evaluate_query_method(gt_pii, unified, q.get("subtype", ""))
                per_query.append({
                    "case_id": cid,
                    "ablation": ab,
                    "method": m,
                    "language": r.get("language", ""),
                    "subtype": r.get("subtype", ""),
                    **metrics,
                })

    print(f"Built per-query metrics: {len(per_query)} rows\n")

    # ===== Aggregate =====
    agg = aggregate(per_query)

    # ===== Print headline =====
    print("=== Overall micro-recall ===")
    for (m, ab), s in agg["overall"].items():
        print(f"  {m.upper()}/{ab.upper()}: {fmt_pct(s['recall'])} ({s['hits']}/{s['n_pii_instances']})")
    print()
    print("=== Recall by PII type (E2 only, both methods) ===")
    for pt in GT_TYPES:
        n = agg["recall_table"][("m1", "e2", pt)]["n"]
        if n == 0:
            continue
        r1 = agg["recall_table"][("m1", "e2", pt)]["recall"]
        r2 = agg["recall_table"][("m2", "e2", pt)]["recall"]
        print(f"  {pt:12s} N={n:3d}   M1={fmt_pct(r1)}   M2={fmt_pct(r2)}")
    print()
    print("=== False positives on D-subtype queries ===")
    for (m, ab), s in agg["fp_d"].items():
        print(f"  {m.upper()}/{ab.upper()}: {s['cases_with_fp']}/{s['n_d_queries']} cases with FP "
              f"({s['total_fp_detections']} total)")
    print()

    # ===== Write outputs =====
    write_per_query_csv(per_query, OUT_PER_QUERY)
    write_summary_csv(agg, OUT_SUMMARY)
    write_report(agg, per_query, OUT_REPORT)

    print(f"wrote {OUT_PER_QUERY.relative_to(ROOT.parent)}")
    print(f"wrote {OUT_SUMMARY.relative_to(ROOT.parent)}")
    print(f"wrote {OUT_REPORT.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
