import json
import random
import time
from datetime import datetime, timedelta
import requests

INPUT_FILE = "dataset.json"
OUTPUT_FILE = "augmented_test_set.json"
SAVE_INTERVAL = 10  # Save every 10 examples
MAX_RETRIES = 5
RETRY_DELAY = 3  # Seconds

# REPLACE THIS WITH YOUR GEMINI API KEY!!
API_KEY = "YOUR-GEMINI-API-KEY"

def map_truth(orig_truth: str):
    if orig_truth in ["true", "mostly-true", "half-true"]:
        return "true"
    elif orig_truth in ["mostly-false", "false", "pants-fire"]:
        return "false"
    
    print(f"WARNING!!! Found a record with unknown truthfulness: {orig_truth}")

def modify_truthfulness(change_type: str, old_truthfulness):
    if change_type == "fabricate":
        return "unknown"
    
    if old_truthfulness == "false":
        if change_type == "contradict":
            return "true"
        return "false"
    
    if old_truthfulness == "true":
        if change_type == "paraphrase":
            return "true"
        return "false"

# Placeholder for Gemini 2.0 Flash API
def call_gemini_model(prompt: str, model: str="2.0-flash-lite"):
    headers = {
        "Content-Type": "application/json"
    }
    if API_KEY == "YOUR-GEMINI-API-KEY":
        raise ValueError(f"Update the code with your API Key!")
    
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

            # Extract response safely
            if "candidates" in response_json and response_json["candidates"]:
                return response_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "Error: API response is missing expected data."
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return f"Error: Unexpected response format - {str(e)}"
    else:
        raise ValueError("Rate Limit!")

def get_augmented_statement(original_statement, mode):
    system_instruction = "ONLY return the new statement. Do not include any other text, explanation or justifications."
    instructions = {
        "paraphrase": "Paraphrase this statement while preserving its meaning:",
        "contradict": "Write a new statement that directly contradicts this:",
        "change_detail": "Modify a small detail in this statement (like a number or name):",
        "fabricate": "Make up a completely new statement about a different topic:"
    }

    prompt = instructions[mode] + f"\n\n'{original_statement}'\n{system_instruction}"
    
    # Retry with backoff
    for attempt in range(MAX_RETRIES):
        try:
            response = call_gemini_model(prompt)
            return response
        except Exception as e:
            print(f"Gemini call failed: {e}, retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY * (2 ** attempt))
    return None  # Failed all attempts

def generate_augmented_test_set():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    augmented = []
    modes = ["skip", "paraphrase", "contradict", "change_detail", "fabricate"]
    weights = [0.95, 0.02, 0.01, 0.01, 0.01]

    for idx, entry in enumerate(data):
        mode = random.choices(modes, weights)[0]
        if mode == "skip":
            continue

        original = entry["statement"]
        augmented_text = get_augmented_statement(original, mode)
        print("")
        print(f"Original Statement: {original}")
        print(f"Mode of change: {mode}")
        print(f"Augmented statement: {augmented_text}")
        print(f"")
        if augmented_text is None:
            continue  # skip if Gemini failed

        # Simulate query time: statement_date + random(1,100) days
        try:
            base_date = datetime.strptime(entry["statement_date"], "%m/%d/%Y")
        except Exception as e:
            print(f"Date parsing failed on entry {idx}: {e}")
            continue

        delta_days = random.randint(1, 100)
        query_date = base_date + timedelta(days=delta_days)
        query_date_str = query_date.strftime("%Y-%m-%d")

        new_truthfulness = modify_truthfulness(mode, map_truth(entry['verdict']))

        augmented.append({
            "augmented_statement": augmented_text,
            "generation_mode": mode,
            "time_stamp": query_date_str,
            "truth_label": new_truthfulness,
            "original_id": idx
        })

        if len(augmented) % SAVE_INTERVAL == 0:
            with open(OUTPUT_FILE, "w") as out_f:
                json.dump(augmented, out_f, indent=2)
            print(f"[Progress] Saved {len(augmented)} entries.")

    # Final save
    with open(OUTPUT_FILE, "w") as out_f:
        json.dump(augmented, out_f, indent=2)
    print("[Done] All augmented entries saved.")

if __name__ == "__main__":
    generate_augmented_test_set()
