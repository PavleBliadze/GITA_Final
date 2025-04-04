import os
import json
import faiss
import numpy as np
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Make sure it is defined in .env")

INDEX_PATH = Path(__file__).parents[1] / "data" / "faiss_index.index"
META_PATH = Path(__file__).parents[1] / "data" / "metadata.json"
EMBEDDING_API = "http://localhost:8000/embed"  

try:
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    index = faiss.read_index(str(INDEX_PATH))
except Exception as e:
    print(f"Error loading metadata or index: {str(e)}")
    metadata = []
    index = None

def get_embedding(code: str):
    response = requests.post(EMBEDDING_API, json={"code": code})
    if response.status_code == 200:
        return np.array(response.json()["embedding"], dtype="float32")
    else:
        raise RuntimeError(f"Embedding failed: {response.text}")

def search_similar(embedding, top_k=5):
    if index is None:
        return []
    D, I = index.search(np.array([embedding]), top_k)
    results = [metadata[i] for i in I[0] if i < len(metadata)]
    return results

def ask_llm(user_code: str, similar_chunks: list):
    context = "\n\n".join([chunk["chunk"] for chunk in similar_chunks]) if similar_chunks else "No similar code found."
    prompt = f"""Your task is to check whether the given code snippet is plagiarized.
User's code:
\"\"\"
{user_code}
\"\"\"

Similar codes from the vector database:
\"\"\"
{context}
\"\"\"

Analyze the codes in detail and compare them.
Then return only one full word:
კი — if the code is plagiarized (similarity is more than 70%)
არა — if the code is not plagiarized (similarity is less than 70%)

The response must be only one word, კი or არა!
"""

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        payload = {
            "model": "gpt-3.5-turbo",  
            "messages": [
                {"role": "system", "content": "You are an expert in detecting code plagiarism. Your response must be either 'კი' or 'არა'."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 5,
            "temperature": 0.0  
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            raw_result = response.json()["choices"][0]["message"]["content"].strip()
            print(f"Raw LLM response: '{raw_result}'")  
            
            if "კი" in raw_result:
                return "კი"
            elif "არა" in raw_result:
                return "არა"
            else:
                print(f"Unexpected LLM response: '{raw_result}' — defaulting to 'არა'")
                return "არა"
        else:
            print(f"OpenAI API error: {response.status_code} - {response.text}")
            return "არა"

    except Exception as e:
        print(f"LLM Error: {str(e)}")
        return "არა"

def check_plagiarism(code: str):
    try:
        embedding = get_embedding(code)
        similar_chunks = search_similar(embedding)
        verdict = ask_llm(code, similar_chunks)
        references = list({chunk["file"] for chunk in similar_chunks})
        return verdict, references
    except Exception as e:
        print(f"Plagiarism check error: {str(e)}")
        return "Error", []