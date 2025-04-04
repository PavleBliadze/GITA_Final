# GITA_Final
Code Plagiarism Detection System
## Aim of the Project
The Code Plagiarism Detection System aims to identify plagiarized code by comparing user-submitted code against a database of code snippets from publicly available GitHub repositories. The system uses a combination of vector embeddings and LLM to determine the similarity between code snippets and detect plagiarism.

Key Components:

Embeddings: Transforms code snippets into numerical vector representations to enable similarity comparisons.

Indexing: Stores embeddings in a FAISS (Facebook AI Similarity Search) index, grouping codes by programming language for optimized, rapid retrieval of similar snippets from a large database.

Plagiarism Detection: Integrates FAISS-based vector search (for fast similarity matching) with a GPT-based LLM (for contextual understanding) to determine if the submitted code is plagiarized.

Evaluation: 
Assesses performance using three strategies:
RAG-only: Retrieval-Augmented Generation, relying solely on vector similarity.
LLM-only: Pure language model analysis without vector search. 
Full System (RAG + LLM): Combines retrieval and LLM for hybrid detection.

## Embedding Logic
The CodeBERT model is used to generate the embeddings. This model is a transformer-based architecture that captures both syntax and semantics of code.

Embedding Overview:

Code Input: The user submits code as a string to the system.

Tokenization: The code is tokenized using the CodeBERT tokenizer, which splits the code into sub-tokens.

Embedding: The tokenized code is passed through the CodeBERT model, which generates a vector representing the code’s semantics.

Vector Storage: The generated vector is then used for similarity searches or further processing.

# How to Run the Embedding Service
Activate the virtual environment:

On Windows:

bash
.\venv\Scripts\activate

Install the required packages for the embedding service:

bash
pip install -r embedding_service/requirements.txt

Run the Embedding Service:

bash
uvicorn plagiarism_checker.embedding_service.main:app --reload --port 8001

This will start the embedding service on http://localhost:8000.

## Indexing Logic
Indexing Process:
Cloning Repositories: The system clones a set of GitHub repositories to collect code files.

Cleaning Code: The code is stripped of comments and unnecessary lines to focus on the core logic.

Chunking: The code is split into logical chunks (based on function or method boundaries).

Generating Embeddings: The chunks of code are embedded using the embedding service.

Creating FAISS Index: The embeddings are stored in a FAISS index for fast similarity searches.

Metadata: Metadata, such as the original file path and chunk ID, is saved for reference during searches.

# How to Run the Indexer
Install Dependencies:
pip install -r indexer/requirements.txt

Run the Indexer:
python indexer/indexer.py

## How the System Works
System Overview
The system works by using FAISS for fast vector similarity searches and GPT-based language models for final plagiarism verification.

User Code Submission: The user submits a code snippet to the Plagiarism API via the /check endpoint.

Embedding Generation: The code is passed to the embedding service, which generates an embedding vector.

FAISS Search: The generated embedding is used to query the FAISS index for similar code snippets from the preprocessed repository code.

Plagiarism Verification:

The system retrieves the top-k similar code snippets (5 in our case) and passes them to the LLM (GPT 3.5-turbo in our case).

The LLM analyzes whether the user’s code is plagiarized based on the similarity results and returns either "კი" (yes) or "არა" (no).

Result: The system returns the verdict and a list of similar code files (if any).

# How to Run the Plagiarism API
Install Dependencies for the API:
pip install -r api/requirements.txt

Start the Plagiarism API:
uvicorn plagiarism_checker.api.main:app --reload --port 8001


How the System Was Evaluated
The system was evaluated using three different detection strategies:

RAG-only (Vector Search): The system only uses the FAISS vector search to find similar code snippets without involving the LLM.

LLM-only (Language Model): The system uses GPT to evaluate plagiarism without using the FAISS vector search.

Full System: Combines both RAG (vector search) and LLM (plagiarism confirmation).

## Creation of Test Cases
The test cases were created to evaluate the system's performance using both plagiarized and non-plagiarized code examples. 

Plagiarized Test Cases:

Code snippets were selected from publicly available GitHub repositories that had known plagiarized code or very similar code snippets.
These code snippets were then inserted into the test_cases.json file with a label of "კი" (yes, plagiarized).

Non-Plagiarized Test Cases:

For each non-plagiarized case, original code snippets were created that did not match any code from the GitHub repositories.
These snippets were added to the test cases with a label of "არა" (no, not plagiarized).

## How to Run the Full System
docker-compose build
docker-compose up

test the system at http://localhost:8001.

stop the system: 
docker-compose down

For cloning this GitHub repository write:
git clone https://github.com/PavleBliadze/GITA_Final.git
