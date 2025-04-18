import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

from retriever import retrieve_with_timestamp, get_vector_db
from langchain_huggingface import HuggingFaceEmbeddings
from reranker import rerank_by_time_and_relevance

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# query = "Most Americans have committed crimes worthy of prison time"
# query_timestamp = to_unix_timestamp("12/8/2014")

db = get_vector_db()
# retrieved_docs = retrieve_with_timestamp(query, before_time=query_timestamp, db=db, k=3)

# reranked_docs = rerank_by_time_and_relevance(query, query_timestamp, retrieved_docs, embedding_model, alpha=0.7, lmbda=0.5)

# Main evaluation and plotting function
def evaluate_grid_search(validation_file="validation_set.json"):
    with open(validation_file, "r") as f:
        val_data = json.load(f)[:100]

    alpha_vals = np.linspace(0.0, 1.0, 11)
    lambda_vals = np.linspace(0.0, 1.0, 11)

    top1_results = np.zeros((len(alpha_vals), len(lambda_vals)))
    top3_results = np.zeros((len(alpha_vals), len(lambda_vals)))

    top1_correct = {}
    top3_correct = {}
    for entry in tqdm(val_data):
        query = entry["augmented_statement"]
        query_timestamp = entry["time_stamp"]
        retrieved_docs = retrieve_with_timestamp(query, before_time=query_timestamp, db=db, k=20)
        for i, alpha in enumerate(alpha_vals):
            for j, lambda_ in enumerate(lambda_vals):
                if (alpha, lambda_) not in top1_correct:
                    top1_correct[(alpha, lambda_)] = 0
                    top3_correct[(alpha, lambda_)] = 0

                reranked_docs = rerank_by_time_and_relevance(query, query_timestamp, retrieved_docs, embedder=embedding_model, alpha=alpha, lmbda=lambda_)
                true_id = entry["original_id"]

                if reranked_docs:
                    if reranked_docs[0].metadata['doc_id'] == true_id:
                        top1_correct[(alpha, lambda_)] += 1
                    for doc in reranked_docs:
                        if doc.metadata['doc_id'] == true_id:
                            top3_correct[(alpha, lambda_)] += 1
                            break

                total = len(val_data)
                top1_results[i, j] = (top1_correct[(alpha, lambda_)] / total) * 100
                top3_results[i, j] = (top3_correct[(alpha, lambda_)] / total) * 100

    # Plotting
    plot_heatmap(alpha_vals, lambda_vals, top1_results, "Top-1 Recall (%)", "top1_recall_heatmap.png")
    plot_heatmap(alpha_vals, lambda_vals, top3_results, "Top-3 Recall (%)", "top3_recall_heatmap.png")

def plot_heatmap(alphas, lambdas, results, title, filename):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    c = ax.imshow(results, origin="lower", cmap="viridis", extent=[lambdas[0], lambdas[-1], alphas[0], alphas[-1]], aspect='auto')
    plt.colorbar(c, ax=ax, label="Recall (%)")
    plt.title(title)
    plt.xlabel("Lambda (recency weight)")
    plt.ylabel("Alpha (similarity weight)")
    ax.set_xticks(lambdas)
    ax.set_yticks(alphas)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

evaluate_grid_search("datasets/validation_set.json")