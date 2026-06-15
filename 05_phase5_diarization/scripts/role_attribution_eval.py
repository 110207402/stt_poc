"""
A2 Role Attribution evaluation — agent vs customer auto-classification.

Task: Given diarization output (speaker A, speaker B segments) without role
labels, predict which speaker is the agent vs the customer.

Baselines:
  B1  rule-based   — keyword detection in transcripts
  B2  text LLM     — Azure OpenAI gpt-4o-mini zero-shot per turn
  B3  acoustic     — pyannote embedding + voice prior from enrollment pool
  B4  fusion       — weighted vote across B1/B2/B3

Eval set: 37 closed-set 2-speaker dialogs from role_attribution split.
         (3 impostor_unknown / 3-speaker cases excluded for closed-set MVP)

Stratified by role_signal_strength: strong (15) / medium (8) / weak (14).
"""

from __future__ import annotations
import json
import csv
import pathlib
import re
import collections
import statistics
import time
import os
from dataclasses import dataclass
from typing import Optional

V2_ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC_PATH = V2_ROOT / "data" / "dialog_specs" / "all_dialogs.json"
RESULTS_DIR = V2_ROOT / "results" / "role_attribution"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# Data loading
# ================================================================

def load_role_attr_cases():
    """Load 40 role_attribution cases, filter to closed-set 2-speaker."""
    all_cases = json.loads(SPEC_PATH.read_text())
    role_cases = [c for c in all_cases if c["split"] == "role_attribution"]
    # MVP: closed-set 2-speaker only
    closed_set = [c for c in role_cases if len(c["participants"]) == 2]
    excluded = [c for c in role_cases if len(c["participants"]) != 2]
    return closed_set, excluded, all_cases


# ================================================================
# B1: Rule-based — keyword pattern matching on transcripts
# ================================================================

# Patterns that strongly suggest the SPEAKER is the agent (not customer)
AGENT_PATTERNS = [
    r"凱基人壽",                    # company self-identification
    r"敝姓",                        # surname self-intro (agent script)
    r"客服中心",
    r"我是.*客服",
    r"請問需要.*服務",
    r"請提供.*身分證",
    r"請報.*身分證",
    r"請告知.*身分證",
    r"請提供.*姓名",
    r"請問您.*姓名",
    r"請問您.*貴姓",
    r"我先確認",
    r"我幫您查",
    r"我這邊.*查",
    r"請您稍.*等",
    r"請稍等",
    r"系統.*顯示",
    r"我們系統",
    r"麻煩您",                       # service-tone
    r"按怎共.*服務",                 # hokkien greeting
    r"我問你.*身分證",
    r"請報你的名",
]

# Patterns that strongly suggest the SPEAKER is the customer
CUSTOMER_PATTERNS = [
    r"我想.*申請",
    r"我想.*問",
    r"我要.*問",
    r"我要.*查",
    r"我的保單",
    r"我的.*理賠",
    r"我.*發生",
    r"我.*受傷",
    r"我.*住院",
    r"我先生",
    r"我太太",
    r"我兒子",
    r"我女兒",
    r"我媽媽",
    r"我爸爸",
    r"麻煩.*你",                     # asking
    r"拜託",
    r"啊我.*的保",                   # hokkien customer phrasing
]


def score_turn_rule_based(text: str) -> tuple[int, int]:
    """Return (agent_signal_count, customer_signal_count) for a single turn."""
    a, c = 0, 0
    for pat in AGENT_PATTERNS:
        if re.search(pat, text): a += 1
    for pat in CUSTOMER_PATTERNS:
        if re.search(pat, text): c += 1
    return a, c


def b1_predict(case: dict) -> dict:
    """Predict which spec speaker is agent using rule scoring on their turn texts.

    Returns: {speaker_label: predicted_role, ...} for 'agent' and 'customer' speakers.
    Plus 'predicted_agent_label' for which spec label was predicted as agent.
    """
    # In our spec, speakers are already labeled. We pretend we don't know which is which
    # and try to recover it from text alone.
    by_speaker = collections.defaultdict(lambda: [0, 0])  # speaker → [agent_score, customer_score]
    for t in case["turns"]:
        spk = t["speaker"]
        if spk == "other":
            continue
        a, c = score_turn_rule_based(t["text"])
        by_speaker[spk][0] += a
        by_speaker[spk][1] += c

    # Aggregate to per-speaker: prefer the speaker with HIGHER agent_score - customer_score
    # as the "agent"
    speakers = list(by_speaker.keys())
    if len(speakers) < 2:
        return {"predicted_agent_label": speakers[0] if speakers else None,
                "confidence": 0.0}
    s0, s1 = speakers
    delta0 = by_speaker[s0][0] - by_speaker[s0][1]
    delta1 = by_speaker[s1][0] - by_speaker[s1][1]
    pred = s0 if delta0 > delta1 else s1
    conf = abs(delta0 - delta1) / max(1, sum(by_speaker[s0]) + sum(by_speaker[s1]))
    return {
        "predicted_agent_label": pred,
        "confidence": round(conf, 3),
        "s0_agent_score": by_speaker[s0][0],
        "s0_customer_score": by_speaker[s0][1],
        "s1_agent_score": by_speaker[s1][0],
        "s1_customer_score": by_speaker[s1][1],
    }


# ================================================================
# B2: Text LLM — Azure OpenAI gpt-4o-mini zero-shot
# ================================================================

def b2_predict(case: dict, llm_client=None) -> dict:
    """Use Azure OpenAI gpt-4o-mini to classify which speaker is the agent.

    Sends ALL turns at once with a structured prompt asking which speaker label
    (the actual labels: 'agent' or 'customer' in spec — but we mask them as 'A' and 'B'
    so LLM doesn't cheat from label name) is the customer service agent.
    """
    if llm_client is None:
        return {"predicted_agent_label": None, "confidence": 0.0, "error": "no llm_client"}

    # Build masked dialog: relabel agent→'A', customer→'B' to hide ground truth
    speakers_present = [t["speaker"] for t in case["turns"] if t["speaker"] != "other"]
    distinct = list(dict.fromkeys(speakers_present))  # preserve order
    if len(distinct) != 2:
        return {"predicted_agent_label": None, "confidence": 0.0, "error": "not_2_speakers"}

    # Random-ish mapping: just preserve order of appearance
    mask = {distinct[0]: "A", distinct[1]: "B"}
    reverse_mask = {v: k for k, v in mask.items()}

    transcript = "\n".join(
        f"[{mask[t['speaker']]}]: {t['text']}"
        for t in case["turns"] if t["speaker"] != "other"
    )

    prompt = f"""這是一段保險公司客服電話的對話逐字稿，兩個說話者標記為 [A] 跟 [B]。

請判斷哪一位是客服人員 (customer service agent)，哪一位是來電客戶 (customer)。

判斷依據：
- 客服通常會說公司名（凱基人壽）、自我介紹敝姓、請對方提供身分證、解釋條款流程
- 客戶通常會表達需求（我想申請/問/查）、提到自己的狀況（住院、理賠、保單問題）

對話：
{transcript}

請只輸出一行 JSON，格式：
{{"agent_speaker": "A" 或 "B", "confidence": 0.0~1.0, "reason": "簡短理由"}}

如果完全無法判斷，"agent_speaker" 填 "uncertain"。"""

    try:
        response = llm_client.chat.completions.create(
            model=os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Extract JSON
        match = re.search(r"\{[^}]+\}", raw)
        if not match:
            return {"predicted_agent_label": None, "confidence": 0.0, "raw_response": raw[:200]}
        parsed = json.loads(match.group(0))
        pred_letter = parsed.get("agent_speaker", "uncertain")
        if pred_letter == "uncertain":
            return {"predicted_agent_label": None, "confidence": 0.0, "reason": parsed.get("reason", "")[:100]}
        if pred_letter not in reverse_mask:
            return {"predicted_agent_label": None, "confidence": 0.0, "error": f"unknown_letter {pred_letter}"}
        return {
            "predicted_agent_label": reverse_mask[pred_letter],
            "confidence": float(parsed.get("confidence", 0.5)),
            "reason": parsed.get("reason", "")[:100],
        }
    except Exception as e:
        return {"predicted_agent_label": None, "confidence": 0.0, "error": f"{type(e).__name__}: {str(e)[:200]}"}


# ================================================================
# B3: Acoustic — voice prior from enrollment pool
# ================================================================

def b3_predict(case: dict, voice_prior: dict) -> dict:
    """Use voice_prior dict (voice_id → agent_likelihood from training data).

    For each spec speaker, look up their voice_id's agent_likelihood and pick
    the speaker with higher likelihood as agent.

    Caveat: in synthetic data, same voices used for both roles, so this is a
    weak signal (voice prior, not voice discriminator).
    """
    parts = case["participants"]
    if "agent" not in parts or "customer" not in parts:
        return {"predicted_agent_label": None, "confidence": 0.0}

    a_voice = parts["agent"]["voice_id"]
    c_voice = parts["customer"]["voice_id"]

    a_prior = voice_prior.get(a_voice, 0.5)  # default 50% if unknown
    c_prior = voice_prior.get(c_voice, 0.5)

    pred = "agent" if a_prior > c_prior else "customer"
    conf = abs(a_prior - c_prior)
    return {
        "predicted_agent_label": pred,
        "confidence": round(conf, 3),
        "agent_voice_prior": round(a_prior, 3),
        "customer_voice_prior": round(c_prior, 3),
    }


def build_voice_prior(all_cases) -> dict:
    """From dev_smoke + dev_tune, compute P(voice serves as agent | voice was used).

    voice_id → agent_likelihood ∈ [0, 1]
    """
    # Count how many times each voice_id is used as agent vs customer
    voice_counts = collections.defaultdict(lambda: [0, 0])  # [agent_count, customer_count]
    for c in all_cases:
        if c["split"] not in ("dev_smoke", "dev_tune"):
            continue
        for role, p in c["participants"].items():
            vid = p.get("voice_id")
            if not vid: continue
            if role == "agent":
                voice_counts[vid][0] += 1
            elif role == "customer":
                voice_counts[vid][1] += 1
    voice_prior = {}
    for vid, (a, c) in voice_counts.items():
        total = a + c
        voice_prior[vid] = a / total if total > 0 else 0.5
    return voice_prior


# ================================================================
# B4: Fusion — weighted vote
# ================================================================

def b4_predict(b1_result, b2_result, b3_result, weights=(0.30, 0.55, 0.15)) -> dict:
    """Weighted soft vote across baselines.

    Default weights: text LLM strongest (0.55), rules (0.30), acoustic (0.15)
    based on expectation that text is most informative for synthetic data.
    """
    scores = {}  # speaker label → cumulative score
    w_rule, w_text, w_acoustic = weights

    for result, weight in [(b1_result, w_rule), (b2_result, w_text), (b3_result, w_acoustic)]:
        pred = result.get("predicted_agent_label")
        conf = result.get("confidence", 0.0)
        if pred is None:
            continue
        scores[pred] = scores.get(pred, 0.0) + weight * conf

    if not scores:
        return {"predicted_agent_label": None, "confidence": 0.0}
    pred = max(scores.keys(), key=lambda k: scores[k])
    total = sum(scores.values())
    conf = scores[pred] / total if total > 0 else 0.0
    return {
        "predicted_agent_label": pred,
        "confidence": round(conf, 3),
        "score_breakdown": {k: round(v, 3) for k, v in scores.items()},
    }


# ================================================================
# Main eval loop
# ================================================================

def evaluate(closed_set, all_cases, use_llm: bool = True):
    voice_prior = build_voice_prior(all_cases)
    print(f"Voice prior (P(voice = agent) from dev_smoke + dev_tune):")
    for v, p in sorted(voice_prior.items()):
        print(f"  {v:30}  P(agent)={p:.2%}")
    print()

    # Initialize Azure OpenAI client
    llm_client = None
    if use_llm:
        try:
            from openai import AzureOpenAI
            llm_client = AzureOpenAI(
                api_key=os.environ.get("AZURE_API_KEY"),
                api_version=os.environ.get("AZURE_API_VERSION", "2024-08-01-preview"),
                azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
            )
            print(f"✓ Azure OpenAI client ready (deployment={os.environ.get('AZURE_CHAT_DEPLOYMENT', 'gpt-4o-mini')})")
        except Exception as e:
            print(f"✗ Azure OpenAI init failed: {e}")
            llm_client = None

    results = []
    print(f"\nEvaluating {len(closed_set)} closed-set cases...\n")

    for i, c in enumerate(closed_set, 1):
        cid = c["case_id"]
        sig = c["role_signal_strength"]
        slc = c["slice"]
        rec = c["recording_start"]

        b1 = b1_predict(c)
        b2 = b2_predict(c, llm_client=llm_client)
        b3 = b3_predict(c, voice_prior=voice_prior)
        b4 = b4_predict(b1, b2, b3)

        # Ground truth: agent label is always 'agent'
        truth = "agent"

        b1_correct = b1.get("predicted_agent_label") == truth
        b2_correct = b2.get("predicted_agent_label") == truth
        b3_correct = b3.get("predicted_agent_label") == truth
        b4_correct = b4.get("predicted_agent_label") == truth

        result_row = {
            "case_id": cid,
            "slice": slc,
            "signal_strength": sig,
            "recording_start": rec,
            "b1_pred": b1.get("predicted_agent_label"),
            "b1_correct": b1_correct,
            "b1_conf": b1.get("confidence", 0),
            "b2_pred": b2.get("predicted_agent_label"),
            "b2_correct": b2_correct,
            "b2_conf": b2.get("confidence", 0),
            "b3_pred": b3.get("predicted_agent_label"),
            "b3_correct": b3_correct,
            "b3_conf": b3.get("confidence", 0),
            "b4_pred": b4.get("predicted_agent_label"),
            "b4_correct": b4_correct,
            "b4_conf": b4.get("confidence", 0),
        }
        results.append(result_row)
        if i % 5 == 0 or i == len(closed_set):
            print(f"  [{i:2}/{len(closed_set)}] {cid} ({sig}/{slc}) — B1={b1_correct} B2={b2_correct} B3={b3_correct} B4={b4_correct}")

        if use_llm and llm_client:
            time.sleep(0.3)  # rate limit guard

    return results, voice_prior


def summarize(results: list[dict]):
    """Compute accuracy overall + stratified by signal_strength."""
    print("\n" + "=" * 78)
    print("Role Attribution — accuracy summary")
    print("=" * 78)

    n = len(results)
    print(f"\nOverall accuracy ({n} cases):")
    for b in ["b1", "b2", "b3", "b4"]:
        correct = sum(1 for r in results if r[f"{b}_correct"])
        labeled = sum(1 for r in results if r[f"{b}_pred"] is not None)
        acc = correct / n if n > 0 else 0
        print(f"  {b.upper():4} {acc:.1%}  ({correct}/{n} correct, {labeled}/{n} labeled)")

    print("\nStratified by signal_strength:")
    by_sig = collections.defaultdict(list)
    for r in results:
        by_sig[r["signal_strength"]].append(r)
    for sig in ["strong", "medium", "weak"]:
        rows = by_sig.get(sig, [])
        if not rows: continue
        print(f"\n  {sig.upper()} ({len(rows)} cases):")
        for b in ["b1", "b2", "b3", "b4"]:
            correct = sum(1 for r in rows if r[f"{b}_correct"])
            print(f"    {b.upper():4} {correct/len(rows):.1%}  ({correct}/{len(rows)})")

    print("\nStratified by recording_start:")
    by_rec = collections.defaultdict(list)
    for r in results:
        by_rec[r["recording_start"]].append(r)
    for rec in ["full_call", "mid_call"]:
        rows = by_rec.get(rec, [])
        if not rows: continue
        print(f"\n  {rec} ({len(rows)} cases):")
        for b in ["b1", "b2", "b3", "b4"]:
            correct = sum(1 for r in rows if r[f"{b}_correct"])
            print(f"    {b.upper():4} {correct/len(rows):.1%}")


def main():
    from dotenv import load_dotenv
    load_dotenv(V2_ROOT.parent.parent / ".env")

    closed_set, excluded, all_cases = load_role_attr_cases()
    print(f"Loaded {len(closed_set)} closed-set 2-speaker cases (excluded {len(excluded)} 3-speaker)")

    results, voice_prior = evaluate(closed_set, all_cases, use_llm=True)
    summarize(results)

    # Save
    out_csv = RESULTS_DIR / "role_attribution_eval.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n→ Per-case results: {out_csv}")

    # Save voice prior for reference
    (RESULTS_DIR / "voice_prior.json").write_text(
        json.dumps(voice_prior, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
