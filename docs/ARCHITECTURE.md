# Architecture and Upgrade Path

The implementation uses SQLite for text chunks and vectors, with an optional Neo4j graph. Without an extraction model, graph edges are heuristic co-occurrence links. The optional extraction model follows explicit stages:

1. Parse documents and preserve page/paragraph locations.
2. Extract entities and typed relationships using a constrained response shape; validate entity membership and evidence against source text.
3. Persist graph entities/edges with provenance in a graph store; parameterize Cypher queries.
4. Generate embeddings for chunks and entities and persist them in a vector index.
5. Retrieve both semantic passages and graph neighborhoods, then deduplicate and rank evidence.
6. Generate an answer only from retrieved evidence, with citations back to document locations.
7. Evaluate entity/relationship precision, retrieval recall, citation accuracy, and multi-hop answer faithfulness.

Do not describe heuristic co-occurrence as a verified relation. The SQLite vector store uses exact scans and suits small corpora; move to a dedicated vector index as scale requires.
