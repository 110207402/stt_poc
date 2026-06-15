"""
Yating TTS - Generate 20 Taiwanese Hokkien insurance dialogues
10 with tai_female_1, 10 with tai_male_1
"""
import requests
import base64
import json
import pathlib
import time

API_KEY = "768b22d585833fbfb1409769fb58490a5c771f90"
ENDPOINT = "https://tts.api.yating.tw/v2/speeches/short"
OUT_DIR = pathlib.Path(__file__).parent / "data" / "audio_hokkien"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    # --- tai_female_1 (h0001~h0010) ---
    ("h0001", "tai_female_1", "claim",
     "你好，我想要問一下我的保單理賠要怎麼申請，上個月我有去住院，診斷書都有，不知道要準備什麼資料。"),
    ("h0002", "tai_female_1", "premium",
     "我想確認一下下個月的保費繳費日期，不知道有沒有變更，我還想問可不可以調整繳費日。"),
    ("h0003", "tai_female_1", "policy",
     "我想要了解一下我的保單附約內容，我有附加一個青春附約，想知道保障的範圍是什麼。"),
    ("h0004", "tai_female_1", "surrender",
     "請問如果我要辦理解約要準備哪些資料，我的保單已經保了十年了，不知道有沒有辦法解約。"),
    ("h0005", "tai_female_1", "beneficiary",
     "我想要變更受益人，請問要怎麼辦理，我父親最近過世了，我想把受益人改成我的妹妹。"),
    ("h0006", "tai_female_1", "loan",
     "請問我可以用保單借款嗎，大概可以借多少，我最近比較急需要用錢，想知道借款的利息是多少。"),
    ("h0007", "tai_female_1", "address",
     "我要更新我的聯絡地址跟電話，我最近搬家了，新的地址是台北市信義區，電話也換了。"),
    ("h0008", "tai_female_1", "claim_status",
     "我上個月有申請理賠，請問審核進度如何，不知道是不是資料不夠齊全，怎麼還沒核定。"),
    ("h0009", "tai_female_1", "product",
     "我想了解你們的心康泰健康險有什麼保障，我今年四十歲，想要投保一個有包含住院跟手術的健康險。"),
    ("h0010", "tai_female_1", "renewal",
     "我的保單快到期了，請問續保的流程是什麼，不知道續保的保費會不會跟原本一樣，還是要重新計算。"),

    # --- tai_male_1 (h0011~h0020) ---
    ("h0011", "tai_male_1", "claim",
     "你好，我父親上個月確診大腸癌，目前在化學治療，請問我可以申請癌症初次診斷保險金嗎，還有哪些項目可以申請。"),
    ("h0012", "tai_male_1", "ltc",
     "我媽媽因為中風造成行動不便，醫生說符合失能狀況，想詢問怎麼申請長期照顧分期保險金，需要準備哪些文件。"),
    ("h0013", "tai_male_1", "surgery",
     "我上週剛開完刀出院，有投保手術終身保險，想請問手術保險金怎麼申請，手術是在地區醫院做的不是醫學中心。"),
    ("h0014", "tai_male_1", "ci",
     "我最近被診斷出需要長期洗腎，醫院已經發給我重大傷病卡，請問這樣可以直接申請重大傷病給付嗎。"),
    ("h0015", "tai_male_1", "accident",
     "我兒子騎腳踏車跌倒造成手腕骨折，在醫院住院治療，請問這算意外傷害可以申請傷害保險金嗎，骨折有特別的給付標準嗎。"),
    ("h0016", "tai_male_1", "maturity",
     "我的儲蓄保險今年到期了，請問滿期金要怎麼領取，是直接匯到帳戶還是需要填什麼申請書。"),
    ("h0017", "tai_male_1", "terms",
     "我想確認一下我的保單條款，上面有一條關於等待期的規定，我不太清楚等待期是從投保日還是從生效日開始算。"),
    ("h0018", "tai_male_1", "death",
     "我太太最近不幸過世了，她有在貴公司投保，請問身故保險金的申請流程是什麼，需要準備哪些文件。"),
    ("h0019", "tai_male_1", "reinstatement",
     "我之前有一張保單因為忘記繳費被停效了，請問可以辦理復效嗎，停效大概三個月了，復效需要重新體檢嗎。"),
    ("h0020", "tai_male_1", "quote",
     "我想要幫我小孩規劃一個教育保險，他今年五歲，請問有什麼適合的商品，保費大概是多少，可以幫我試算一下嗎。"),
]


def tts_generate(text: str, voice: str) -> bytes:
    headers = {
        "key": API_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "input": {"text": text, "type": "text"},
        "voice": {"model": voice, "speed": 1.0, "pitch": 1.0, "energy": 1.0},
        "audioConfig": {"encoding": "LINEAR16", "sampleRate": "16K"}
    }
    resp = requests.post(ENDPOINT, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return base64.b64decode(data["audioContent"])


def main():
    print(f"Generating {len(CASES)} Hokkien TTS audio files → {OUT_DIR}")
    print(f"10 × tai_female_1  +  10 × tai_male_1\n")

    results = []
    for i, (case_id, voice, category, text) in enumerate(CASES, 1):
        out_path = OUT_DIR / f"{case_id}.wav"
        print(f"  [{i:02d}/{len(CASES)}] {case_id} ({voice}) ...", end=" ", flush=True)
        try:
            audio_bytes = tts_generate(text, voice)
            out_path.write_bytes(audio_bytes)
            size_kb = len(audio_bytes) / 1024
            print(f"OK ({size_kb:.0f} KB) → {out_path.name}")
        except Exception as e:
            print(f"FAIL: {e}")
        results.append({"case_id": case_id, "voice": voice, "category": category, "text": text})
        time.sleep(0.3)  # polite rate limiting

    # Save manifest
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nDone! Manifest saved to {manifest_path}")
    print(f"Files: {list(OUT_DIR.glob('*.wav'))}")


if __name__ == "__main__":
    main()
