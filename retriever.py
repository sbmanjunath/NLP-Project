from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime

import chromadb

from ingest import to_unix_timestamp

# chroma_client = chromadb.PersistentClient(path="chroma_store")
# chroma_client.delete_collection("test")

persistent_client = chromadb.PersistentClient(path="chroma_store")
collection = persistent_client.get_or_create_collection("dataset")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    client=persistent_client,
    collection_name="dataset",
    embedding_function=embedding_model,
)

def retrieve_with_timestamp(query: str, before_time: str, db: Chroma, k: int = 5):
    return db.similarity_search(
        query=query,
        k=k,
        filter={"time_stamp": {"$lt": to_unix_timestamp(before_time)}}
    )

def get_vector_db():
    return db

# results = retrieve_with_timestamp("Most Americans have committed crimes worthy of prison time", before_time="12/8/2014", db=db, k=3)

# print("Retrieved Results:")
# for doc in results:
#     print(f"[{doc.metadata['doc_id']}] {doc.page_content.strip()} (time: {doc.metadata['time_stamp']}) (truth: {doc.metadata['truthfulness']})")
