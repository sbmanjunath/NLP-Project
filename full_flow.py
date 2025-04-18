from retriever import retrieve_with_timestamp, get_vector_db
from langchain_huggingface import HuggingFaceEmbeddings
from reranker import rerank_by_time_and_relevance
from ingest import to_unix_timestamp

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
query = "Most Americans have committed crimes worthy of prison time"
query_timestamp = to_unix_timestamp("12/8/2014")

db = get_vector_db()
retrieved_docs = retrieve_with_timestamp(query, before_time=query_timestamp, db=db, k=3)

reranked_docs = rerank_by_time_and_relevance(query, query_timestamp, retrieved_docs, embedding_model, alpha=0.7, lmbda=0.5)

for d in reranked_docs:
    print(d)

