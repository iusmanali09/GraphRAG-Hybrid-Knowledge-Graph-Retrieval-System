# Local Setup

```sh
cd "GraphRAG-Hybrid-Knowledge-Graph-Retrieval-System"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export GRAPHRAG_EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2'
uvicorn app.main:app --reload --host 127.0.0.1 --port 8020
```

Open `http://127.0.0.1:8020/docs`. Set `GRAPHRAG_DATA_DIR` or `GRAPHRAG_DB_PATH` to change local storage paths.

The API accepts multipart uploads at `POST /documents` (PDF or DOCX) and JSON questions at `POST /query`, for example `{"question":"What does Acme make?","top_k":5}`. The app has no authentication and is intended for local development only. It extracts embedded document text; scanned PDFs need OCR, which is not included.

## Neo4j graph store

```sh
export NEO4J_PASSWORD='use-a-long-random-local-password'
docker compose up -d neo4j
export NEO4J_URI='bolt://127.0.0.1:7687'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='same-password-as-compose'
```

The service uses parameterized 1–2 hop Cypher. Set optional `GRAPHRAG_LLM_URL`, `GRAPHRAG_LLM_MODEL`, and `GRAPHRAG_LLM_API_KEY` for query entity extraction and grounded answer generation. Without them it uses capitalized-phrase extraction and returns evidence without synthesizing an answer. SQLite stores the vectors and fallback graph; exact vector scans are suitable only for small local corpora.
