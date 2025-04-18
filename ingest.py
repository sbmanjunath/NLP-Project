from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from datetime import datetime

import chromadb
import json

# chroma_client = chromadb.PersistentClient(path="chroma_store")
# chroma_client.delete_collection("test")

INGESTION_BATCH_SIZE = 5000

def to_unix_timestamp(ts: str) -> float:
    if isinstance(ts, float):
        return ts
    if "-" in ts:
        return datetime.strptime(ts, "%Y-%m-%d").timestamp()
    return datetime.strptime(ts, "%m/%d/%Y").timestamp()

# --- Step 1: Ingest documents into Chroma ---
def ingest_documents_chroma(docs: List[dict], persist_directory: str = "chroma_store") -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    chroma_db = Chroma(
        collection_name="dataset",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    documents = []

    for i, doc in enumerate(docs):
        if i % 10 == 0:
            print(f"Completed {i} records")
        chunks = splitter.split_text(doc["statement"])
        truthfulness = doc['verdict']

        lc_docs = [
            Document(
                page_content=chunk,
                metadata={
                    "doc_id": i,
                    "time_stamp": to_unix_timestamp(doc["statement_date"]),
                    "chunk_id": it,
                    "truthfulness": truthfulness,
                    "source": doc['statement_originator']
                }
            )
            for it, chunk in enumerate(chunks)
        ]
        documents.extend(lc_docs)
        if len(documents) > INGESTION_BATCH_SIZE:
            print(f"Ingested {INGESTION_BATCH_SIZE} documents")
            chroma_db.add_documents(documents)
            documents = []

    chroma_db.add_documents(documents)

    return chroma_db

# --- Example usage ---
def ingest_dataset(dataset_path: str = "datasets/ground_truth.json"):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    db = ingest_documents_chroma(data)
    print("Ingested dataset!")
    return db

if __name__ == "__main__":
    ingest_dataset()