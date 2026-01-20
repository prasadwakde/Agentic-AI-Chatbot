import os
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ChromaDB initialization
client = chromadb.PersistentClient(
    path="rag_db",
    settings=Settings(anonymized_telemetry=False)
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

# Collections
GLOBAL_COLLECTION_NAME = "global_ai_ml_kb"


def get_global_collection():
    return client.get_or_create_collection(
        name=GLOBAL_COLLECTION_NAME,
        metadata={"scope": "global"}
    )


def get_user_collection(user_id: str):
    return client.get_or_create_collection(
        name=f"user_docs_{user_id}",
        metadata={"user_id": user_id}
    )

# Ingestion
def ingest_text(
    collection,
    source_name: str,
    text: str
) -> int:
    chunks = text_splitter.split_text(text)
    embeddings = embedding_model.embed_documents(chunks)

    ids = [f"{source_name}_{i}" for i in range(len(chunks))]

    metadatas = [
        {
            "source": source_name,
            "chunk_id": i
        }
        for i in range(len(chunks))
    ]

    existing_ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    try:
        collection.delete(ids=existing_ids)
    except Exception:
        pass
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


def ingest_user_document(user_id: str, filename: str, text: str) -> int:
    collection = get_user_collection(user_id)
    return ingest_text(collection, filename, text)


def ingest_global_document(filename: str, text: str) -> int:
    collection = get_global_collection()
    return ingest_text(collection, filename, text)


# Bootstrap global knowledge base
def bootstrap_global_kb(kb_path: str = "knowledge_base"):
    """
    Load all .txt files from knowledge_base/ into the global collection.
    This should be called ONCE at backend startup.
    """
    if not os.path.isdir(kb_path):
        print(f"[RAG] No knowledge_base directory found at {kb_path}")
        return

    collection = get_global_collection()

    for file in os.listdir(kb_path):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(kb_path, file)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = ingest_text(collection, file, text)
        print(f"[RAG] Ingested {file} ({chunks} chunks)")


# Retrieval
def _query_collection(
    collection,
    query: str,
    k: int
) -> List[Tuple[str, dict]]:
    query_embedding = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return list(zip(documents, metadatas))


def rag_search_tool(
    query: str,
    user_id: str | None = None,
    k: int = 3
) -> str:
    """
    Search global AI/ML knowledge base + optional user documents.
    Returns grounded context with sources.
    """
    retrieved = []

    # Search global KB
    global_results = _query_collection(
        get_global_collection(),
        query,
        k
    )
    retrieved.extend(global_results)

    # Search user docs if provided
    if user_id:
        user_results = _query_collection(
            get_user_collection(user_id),
            query,
            k
        )
        retrieved.extend(user_results)

    if not retrieved:
        return "No relevant information found in the knowledge base."

    # Build context with sources
    context_blocks = []
    for doc, meta in retrieved:
        source = meta.get("source", "unknown")
        context_blocks.append(
            f"[SOURCE: {source}]\n{doc}"
        )

    return "\n\n".join(context_blocks)
