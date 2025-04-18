# NLP-Project
NLP Project (COMP 586 Spring 2025) - Time-based Fact-checking and Fake News Detection

## ToDo

1. Classical Retriever: Sagar [DONE]
2. Time-based Re-ranker: Sagar [DONE]
3. LLM-based classification: Param
4. BERT-based classification: Param

## Steps to use retrieval + reranking
1. Install the packages in requirements.txt. You may need C++14 to install chromadb.
2. Run ingest.py. You need to do this ONLY once - this will create the chromadb vector store.
3. Refer to full_flow.py to understand the usage of the retriever and reranker. Optim