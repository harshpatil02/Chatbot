from app.database import create_tables, add_document_chunk
from app.ingestion import embed_text

create_tables()
text = (
    "Postgres with pgvector supports vector search. "
    "HNSW is an efficient graph index for approximate nearest neighbors. "
    "This document mentions BM25 and lexical overlap as fallback retrieval methods."
)
embedding = embed_text(text)
chunk = add_document_chunk(text, metadata={"source": "demo"}, embedding=embedding)
print("ADDED", chunk.id)
