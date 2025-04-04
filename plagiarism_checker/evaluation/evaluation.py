import json
import csv
import requests
from plagiarism_checker.api.detector import get_embedding, search_similar, ask_llm
from pathlib import Path
TEST_SET_PATH = Path(__file__).parent / "test_cases.json"
OUTPUT_CSV = Path(__file__).parent / "results.csv"
SIMILARITY_THRESHOLD = 0.8 
TOP_K = 5

def cosine_sim(a, b):
    a, b = a / (a**2).sum()**0.5, b / (b**2).sum()**0.5
    return (a * b).sum()

def detect_rag_only(code: str):
    try:
        user_emb = get_embedding(code)
        results = search_similar(user_emb, top_k=TOP_K)
        if not results:
            return "არა"

        top_chunk = results[0]["chunk"]
        top_emb = get_embedding(top_chunk)
        score = cosine_sim(user_emb, top_emb)
        return "კი" if score >= SIMILARITY_THRESHOLD else "არა"
    except Exception as e:
        print(f"RAG-only error: {e}")
        return "Error"

def detect_llm_only(code: str):
    try:
        return ask_llm(code, [])
    except Exception as e:
        print(f"LLM-only error: {e}")
        return "Error"

def detect_full_system(code: str):
    try:
        resp = requests.post("http://localhost:8001/check", json={"code": code})
        return resp.json().get("verdict", "Error")
    except Exception as e:
        print(f"Full system error: {e}")
        return "Error"

def main():
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    rows = []
    for item in test_cases:
        code, label = item["code"], item["label"]
        rag = detect_rag_only(code)
        llm = detect_llm_only(code)
        full = detect_full_system(code)

        print(f"\n Test case: expected={label} | rag={rag} | llm={llm} | full={full}")
        rows.append({
            "code": code,
            "label": label,
            "rag_verdict": rag,
            "llm_verdict": llm,
            "full_verdict": full
        })

    with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n Evaluation complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
