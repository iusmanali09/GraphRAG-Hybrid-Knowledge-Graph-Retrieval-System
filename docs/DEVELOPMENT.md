# Development Protocol

- Treat document contents as untrusted input; enforce upload, extraction, and query limits.
- Preserve document provenance and chunk identifiers for every returned evidence item.
- Keep heuristic entity links clearly labeled until validated relation extraction is implemented.
- Add tests for malformed PDFs/DOCX files, empty/scanned documents, Unicode, size limits, chunk boundaries, duplicate uploads, and query ranking.
- If adding a model, separate extraction prompts from documents, validate structured output, and persist model/prompt versions.
- Before multi-tenant or network deployment, add authentication, authorization, tenant scoping, encryption, retention, deletion, and audit requirements.

## Pull request checklist

- [ ] Ingestion limits and provenance remain enforced.
- [ ] Retrieval returns evidence that maps to source chunks.
- [ ] Heuristic edges are not presented as verified facts.
- [ ] Tests cover malformed and boundary inputs.
- [ ] Documentation reflects actual graph/vector capabilities.
