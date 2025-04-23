import requests
import json
import datetime

from retriever import retrieve_with_timestamp, get_vector_db
from langchain_huggingface import HuggingFaceEmbeddings
from reranker import rerank_by_time_and_relevance
from ingest import to_unix_timestamp
import time
from tqdm import tqdm

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def yyyy_mm_dd_to_ddmmyyyy(date):
    splits = date.split("-")
    return f"{splits[2]}/{splits[1]}/{splits[0]}"

def unix_timestamp_to_ddmmyyyy(unix_timestamp):
  """
  Converts a Unix timestamp to a dd/mm/yyyy string.

  Args:
    unix_timestamp: The Unix timestamp (in seconds).

  Returns:
    A string representing the date in dd/mm/yyyy format.
  """
  datetime_object = datetime.datetime.fromtimestamp(unix_timestamp)
  return datetime_object.strftime("%d/%m/%Y")

def get_classification_prompt(claim, date_of_claim, docs):
    prompt = f"""
You are a Fake News Classifier. You are given a claim maid by a user, and you need to predict the truthfulness of the claim. The valid options are:
- true
- false
- insufficient_information

You are given a set of fact-checked claims with their truthfulness. These claims may or may not be related to the user's claim. The fact-checked claim can be treated as the ground truth. All dates are in the format ddm/mm/yyyy.

You should factor in the dates of the user's claim and the fact-checked claims in your decision.

User's claim: {claim}
Date of user's claim: {date_of_claim}
"""
    for doc in docs:
        prompt += f"""
Fact-checked Claim: {doc.page_content}
Date of claim: {unix_timestamp_to_ddmmyyyy(doc.metadata['time_stamp'])}
Truthfulness of claim: {doc.metadata['truthfulness']}

"""
    prompt += """
Your output needs to be exactly one word. If the fact-checked claims support the user's claim, say 'true'. If the fact-checked claims contradict the user's claim, say 'false'. If the fact-checked claims do not offer sufficient information to decide the truthfulness of the user's claim, say 'insufficient_information'.
"""
    return prompt


def get_classification_prompt_cot(claim, date_of_claim, docs):
    prompt = f"""
You are a Fake News Classifier. You are given a claim maid by a user, and you need to predict the truthfulness of the claim. The valid options are:
- true
- false
- insufficient_information

You are given a set of fact-checked claims with their truthfulness. These claims may or may not be related to the user's claim. The fact-checked claim can be treated as the ground truth. All dates are in the format ddm/mm/yyyy.

You should factor in the dates of the user's claim and the fact-checked claims in your decision.

User's claim: {claim}
Date of user's claim: {date_of_claim}
"""
    for doc in docs:
        prompt += f"""
Fact-checked Claim: {doc.page_content}
Date of claim: {unix_timestamp_to_ddmmyyyy(doc.metadata['time_stamp'])}
Truthfulness of claim: {doc.metadata['truthfulness']}

"""
    prompt += """
Start by logically laying out your thoughts. Then, in the last line, respond with one word on the truthfulness of the statement. If the fact-checked claims support the user's claim, say 'true'. If the fact-checked claims contradict the user's claim, say 'false'. If the fact-checked claims do not offer sufficient information to decide the truthfulness of the user's claim, say 'insufficient_information'.
"""
    return prompt

API_KEY = "api-key"

def call_gemini(prompt: str, model: str="2.0-flash"):
    headers = {
        "Content-Type": "application/json"
    }
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-{model}:generateContent?key={API_KEY}"

    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        try:
            response_json = response.json()
            # Debugging: Print API response to inspect structure
            # print(json.dumps(response_json, indent=2))

            # Extract response safely
            if "candidates" in response_json and response_json["candidates"]:
                return response_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "Error: API response is missing expected data."
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return f"Error: Unexpected response format - {str(e)}"
    else:
        return f"Error: {response.status_code}, {response.text}"
    
def parse_llm_into_scenario(llm_response):
    response_alt = llm_response.replace("\n", "")
    response_alt = response_alt.replace("```json", '"')
    response_alt = response_alt.replace("```", "")
    response_alt = response_alt[1:]
    questions = json.loads(response_alt)

    return questions

outputs = []

def get_label_from_llm(response):
    last_word = response.split(" ")[-1]
    if "true" in last_word:
        return "true"
    if "false" in last_word:
        return "false"
    return "unknown"

def run(mode="cot"):
    with open("datasets/test_set.json", "r") as f:
        test_data = json.load(f)[:50]

    y_true = []
    y_pred = []

    for entry in tqdm(test_data):
        query = entry["augmented_statement"]
        query_timestamp = entry["time_stamp"]
        query_timestamp_alt = yyyy_mm_dd_to_ddmmyyyy(entry["time_stamp"])

        db = get_vector_db()
        retrieved_docs = retrieve_with_timestamp(query, before_time=query_timestamp, db=db, k=3)

        reranked_docs = rerank_by_time_and_relevance(query, query_timestamp, retrieved_docs, embedding_model, alpha=0.8, lmbda=0.5)
        # import pdb
        # pdb.set_trace()
        # for d in reranked_docs:
        #     print(d)
        if mode == "cot":
            prompt = get_classification_prompt_cot(query, query_timestamp_alt, reranked_docs)
        else:
            prompt = get_classification_prompt(query, query_timestamp_alt, reranked_docs)

        response = call_gemini(prompt,model="2.0-flash-lite")
        response = response.replace("\n", "")
        new_output = {}
        new_output["question"] = query
        new_output["claim_time"] = query_timestamp_alt
        new_output["llm_response"] = get_label_from_llm(response)
        new_output["ground_truth"] = entry["truth_label"]

        new_output["docs"] = []
        ids = []
        for doc in reranked_docs:
            new_output["docs"].append({"claim": doc.page_content, "time": unix_timestamp_to_ddmmyyyy(doc.metadata['time_stamp']), "truthfulness": doc.metadata['truthfulness']})
            ids.append(doc.metadata['doc_id'])

        original_id = entry["original_id"]

        new_output["is_truth_retrieved"] = True if original_id in ids else False
        new_output["test_mode"] = entry["generation_mode"]

        outputs.append(new_output)
        y_true.append(new_output["ground_truth"])
        y_pred.append(new_output["llm_response"])
        time.sleep(3)

    # Method 1: Using precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    print("Precision per class:", precision)
    print("Recall per class:", recall)
    print("F1-score per class:", f1)
    print("Support per class:", support)

    # Method 2: Using classification_report (more comprehensive)
    report = classification_report(y_true, y_pred)
    print(report)

    # You can access the report as a dictionary if needed
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    print(report_dict)

run()

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(outputs, f, ensure_ascii=False, indent=4)