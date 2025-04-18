from typing import List
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from math import exp

from ingest import to_unix_timestamp

def years_between(ts1: float, ts2: float) -> float:
    return abs(ts1 - ts2) / (365.25 * 24 * 3600)

def rerank_by_time_and_relevance(
    query: str,
    query_timestamp: float,
    retrieved_docs: List[Document],
    embedder: HuggingFaceEmbeddings,
    top_k: int = 3,
    alpha: float = 0.8,
    lmbda: float = 0.5
) -> List[Document]:

    query_vec = embedder.embed_query(query)
    query_ts_unix = to_unix_timestamp(query_timestamp)

    scored_docs = []
    for doc in retrieved_docs:
        doc_vec = embedder.embed_query(doc.page_content)
        semantic_score = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))

        doc_ts_unix = doc.metadata.get("time_stamp")
        if doc_ts_unix is None:
            continue

        time_diff_years = years_between(query_ts_unix, doc_ts_unix)
        recency_score = exp(-lmbda * time_diff_years)

        final_score = alpha * semantic_score + (1 - alpha) * recency_score

        scored_docs.append((final_score, doc))

    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored_docs[:top_k]]
