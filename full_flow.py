from retriever import retrieve_with_timestamp, get_vector_db
from langchain_huggingface import HuggingFaceEmbeddings
from reranker import rerank_by_time_and_relevance
from ingest import to_unix_timestamp
import torch
from transformers import BertTokenizer
from bert_classification import predict, BertForSequenceClassification

def full_pipeline(query, query_timestamp, bert_model_path='outputs/bert_classifier.pt', top_k=3):
    """
    Complete pipeline: retrieval -> reranking -> classification
    """
    # Step 1: Retrieve documents
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = get_vector_db()
    timestamp_unix = to_unix_timestamp(query_timestamp)
    
    retrieved_docs = retrieve_with_timestamp(query, before_time=timestamp_unix, db=db, k=top_k*2)
    print(f"\nRetrieved {len(retrieved_docs)} documents")
    
    # Step 2: Rerank documents
    reranked_docs = rerank_by_time_and_relevance(
        query, 
        timestamp_unix, 
        retrieved_docs, 
        embedding_model, 
        top_k=top_k,
        alpha=0.8, 
        lmbda=0.5
    )
    print(f"\nReranked to {len(reranked_docs)} documents")
    
    # Step 3: Use BERT to classify the claim and provide supporting evidence
    # Load BERT model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,  # Three labels: false (0), unknown (1), true (2)
        output_attentions=False,
        output_hidden_states=False
    )
    model.load_state_dict(torch.load(bert_model_path, map_location=device))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Get classification result
    classification_result = predict(query, model, tokenizer, device)
    
    # Print results
    print(f"\n==== Results for query: '{query}' ====")
    print(f"Date: {query_timestamp}")
    print(f"\nVerdict: {classification_result['prediction']}")
    
    # Print confidence scores for all three classes
    print(f"Confidence: True: {classification_result['confidence']['True']:.4f}, " +
          f"False: {classification_result['confidence']['False']:.4f}, " +
          f"Unknown: {classification_result['confidence']['Unknown']:.4f}")
    
    print("\nSupporting evidence:")
    for i, doc in enumerate(reranked_docs):
        print(f"\n[{i+1}] {doc.page_content}")
        print(f"    Date: {doc.metadata.get('time_stamp')}")
        print(f"    Truth: {doc.metadata.get('truthfulness')}")
    
    return {
        'classification': classification_result,
        'supporting_docs': reranked_docs
    }

if __name__ == "__main__":
    # Example usage
    query = "Most Americans have committed crimes worthy of prison time"
    query_timestamp = "12/8/2014"
    
    result = full_pipeline(query, query_timestamp)

