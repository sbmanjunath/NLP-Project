# NLP-Project
NLP Project (COMP 586 Spring 2025) - Time-based Fact-checking and Fake News Detection

## ToDo

1. Classical Retriever: Sagar [DONE]
2. Time-based Re-ranker: Sagar [DONE]
3. LLM-based classification: Sagar
4. BERT-based classification: Param [DONE]

## Steps to use retrieval + reranking
1. Install the packages in requirements.txt. You may need C++14 to install chromadb.
2. Run ingest.py. You need to do this ONLY once - this will create the chromadb vector store.
3. Refer to full_flow.py to understand the usage of the retriever and reranker.

## Steps to use BERT classification
1. Install additional dependencies with `pip install -r requirements.txt`
2. Train the BERT model by running `python train_bert.py`
   - You can adjust training parameters: `python train_bert.py --epochs 4 --batch_size 32`
   - Training outputs (model, plots, reports) are saved to the `outputs/` directory
3. To use the full pipeline (retrieval → reranking → classification), run `python full_flow.py`

## Model Architecture
The project includes three main components:
1. **Retriever**: Uses ChromaDB to find semantically relevant documents with a timestamp filter
2. **Re-ranker**: Re-ranks documents based on semantic relevance and temporal proximity
3. **BERT Classifier**: Fine-tuned BERT model for three-way classification of statements (true, false, unknown)
4. **LLM Classifier**: 

## Performance
The BERT classifier is trained on the labeled dataset and achieves:
- Three-way classification of statements as true, false, or unknown
- Visualization of training progress and confusion matrices
- Integration with the retrieval pipeline for evidence-based fact checking

## Three-way Classification
The BERT model classifies statements into three categories:
- **True**: Statements that are verified as factually accurate
- **False**: Statements that are verified as factually inaccurate
- **Unknown**: Statements where the factual accuracy cannot be determined with confidence

This comprehensive classification system provides a more nuanced approach to fact-checking, acknowledging that not all statements can be definitively categorized as true or false.

# Fact-Checking Dataset EDA Suite

This repository contains a suite of tools for Exploratory Data Analysis (EDA) of fact-checking datasets. The scripts analyze datasets from the NLP Project focused on fake news detection.

## Datasets

The analysis is performed on the following datasets:
- `/Users/param/Desktop/Original_NLP/NLP-Project/datasets/train_set.json`
- `/Users/param/Desktop/Original_NLP/NLP-Project/datasets/validate_set.json`
- `/Users/param/Desktop/Original_NLP/NLP-Project/datasets/test_set.json`

## Setup and Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Make sure all scripts have execution permissions:

```bash
chmod +x *.py
```
