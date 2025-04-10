import os
import json
import git
import requests
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)

REPO_LIST_FILE = Path(__file__).parent / config["repo_list"]
CLONE_DIR = (Path(__file__).resolve().parent / config["clone_dir"]).resolve()
ALLOWED_EXTENSIONS = config["file_extensions"]

INDEX_PATH = Path(__file__).parents[1] / "data" / "faiss_index.index"
META_PATH = Path(__file__).parents[1] / "data" / "metadata.json"

EMBEDDING_API = "http://localhost:8000/embed"

def get_embedding(code: str):
    response = requests.post(EMBEDDING_API, json={"code": code})
    if response.status_code == 200:
        return np.array(response.json()["embedding"], dtype="float32")
    else:
        raise RuntimeError(f"Embedding failed: {response.text}")

def get_embedding_dimension():
    sample_code = "def sample(): pass"
    sample_embedding = get_embedding(sample_code)
    return len(sample_embedding)

if INDEX_PATH.exists():
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    dimension = get_embedding_dimension()
    print(f"Using embedding dimension: {dimension}")
    index = faiss.IndexFlatL2(dimension)
    metadata = []

def clone_repo(url):
    repo_name = url.strip().split("/")[-1].replace(".git", "")
    dest = CLONE_DIR / repo_name
    if dest.exists():
        print(f"Repo already cloned: {repo_name}")
        return dest
    print(f"📥 Cloning {repo_name}...")
    try:
        git.Repo.clone_from(url, dest)
        return dest
    except Exception as e:
        print(f"Failed to clone {url}: {e}")
        return None

def detect_language(code):
    try:
        lexer = guess_lexer(code)
        return lexer.name.lower()
    except ClassNotFound:
        return "unknown"

def strip_comments(code, ext):
    lines = code.splitlines()
    code_without_single_line_comments = []
    for line in lines:
        line_stripped = line.strip()
        if ext == ".py" and line_stripped.startswith("#"):
            continue
        if ext in [".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp"] and line_stripped.startswith("//"):
            continue
        code_without_single_line_comments.append(line)

    code = "\n".join(code_without_single_line_comments)

    if ext in [".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp"]:
        in_comment = False
        result = []
        i = 0
        while i < len(code):
            if code[i:i+2] == "/*" and not in_comment:
                in_comment = True
                i += 2
            elif code[i:i+2] == "*/" and in_comment:
                in_comment = False
                i += 2
            elif not in_comment:
                result.append(code[i])
                i += 1
            else:
                i += 1
        code = "".join(result)

    if ext == ".py":
        for quote_type in ['"""', "'''"]:
            while True:
                start = code.find(quote_type)
                if start == -1:
                    break
                end = code.find(quote_type, start + 3)
                if end == -1:
                    break
                replacement = "\n" * code[start:end+3].count("\n")
                code = code[:start] + replacement + code[end+3:]

    return code

def logical_chunk(code: str, max_tokens=512):
    chunks = []
    current = []
    for line in code.splitlines():
        current.append(line)
        joined = "\n".join(current)
        if len(joined.split()) >= max_tokens:
            chunks.append(joined)
            current = []
    if current:
        chunks.append("\n".join(current))
    return chunks

def fallback_chunk(code: str, window=256, overlap=64):
    words = code.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+window]
        chunks.append(" ".join(chunk))
        i += window - overlap
    return chunks

def extract_code_files(repo_path):
    code_files = []
    for root, _, files in os.walk(repo_path):
        if any(x in root for x in ["__pycache__", "node_modules", ".git", "test"]):
            continue
        for file in files:
            if any(file.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                code_files.append(Path(root) / file)
    return code_files

def process_file(file_path: Path):
    try:
        ext = file_path.suffix
        code = file_path.read_text(encoding="utf-8", errors="ignore")
        if not code.strip():
            return

        code = strip_comments(code, ext)
        lang = detect_language(code)

        chunks = logical_chunk(code)
        final_chunks = []
        for chunk in chunks:
            if len(chunk.split()) <= 512:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(fallback_chunk(chunk))

        for i, chunk in enumerate(final_chunks):
            embedding = get_embedding(chunk)
            index.add(np.array([embedding]))
            metadata.append({
                "file": str(file_path),
                "ext": ext,
                "lang": lang,
                "chunk_id": i,
                "chunk": chunk
            })
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def main():
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPO_LIST_FILE) as f:
        repo_urls = [line.strip() for line in f if line.strip()]

    for url in repo_urls:
        repo_path = clone_repo(url)
        if not repo_path:
            continue
        print(f"Indexing files in {repo_path}...")
        files = extract_code_files(repo_path)
        for file_path in tqdm(files, desc=f"Indexing {repo_path.name}"):
            process_file(file_path)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Indexing complete.")

if __name__ == "__main__":
    main()
