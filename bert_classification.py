import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from sklearn.utils.class_weight import compute_class_weight

class NewsDataset(Dataset):
    def __init__(self, statements, labels, tokenizer, max_length=128):
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.statements)
    
    def __getitem__(self, idx):
        statement = self.statements[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            statement,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    """
    Load dataset from a JSON file
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    statements = [item['augmented_statement'] for item in data]
    
    # Convert truth labels to three-way classification (true: 2, false: 0, unknown: 1)
    labels = []
    
    for item in data:
        if item['truth_label'] == 'true':
            labels.append(2)
        elif item['truth_label'] == 'false':
            labels.append(0)
        elif item['truth_label'] == 'unknown':
            labels.append(1)
    
    return statements, labels

def train_bert_model(train_dataloader, val_dataloader, device, num_epochs=4, learning_rate=2e-5, 
                    weight_decay=0.01, dropout=0.1, warmup_ratio=0.1, 
                    class_weights=True, gradient_accumulation_steps=2):
    
    # Create model with custom dropout
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,  # Three labels: false (0), unknown (1), true (2)
        output_attentions=False,
        output_hidden_states=False,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout
    )
    
    model.to(device)
    
    # Calculate class weights
    if class_weights:
        # Extract all labels from training set to compute class weights
        all_labels = []
        for batch in train_dataloader:
            all_labels.extend(batch['label'].numpy())
        
        class_weights_vals = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(all_labels),
            y=all_labels
        )
        
        # Convert to tensor and move to device
        class_weights_tensor = torch.tensor(class_weights_vals, dtype=torch.float).to(device)
        print(f"Using class weights: {class_weights_vals}")
    else:
        class_weights_tensor = None
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
    
    # Calculate total training steps and warmup steps
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Create scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    training_stats = []
    best_val_f1 = 0
    best_model = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        total_train_loss = 0
        
        steps_this_epoch = 0
        optimizer.zero_grad()
        
        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
    
            if class_weights_tensor is not None:
                # Create one-hot encoded matrix for labels
                one_hot_labels = torch.zeros(labels.size(0), 3, device=device)
                one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
                
                # Apply weights to logits before computing loss
                weighted_logits = outputs.logits * class_weights_tensor
                loss = torch.nn.functional.cross_entropy(
                    weighted_logits, 
                    labels,
                    reduction='mean'
                )
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            total_train_loss += loss.item() * gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            steps_this_epoch += 1
            if steps_this_epoch % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        predictions = []
        true_labels = []
        
        for batch in tqdm(val_dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            loss = outputs.loss
            total_val_loss += loss.item()
            
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = accuracy_score(true_labels, predictions)
        
        # Calculate F1, precision, recall
        val_report = classification_report(
            true_labels, 
            predictions, 
            target_names=['False', 'Unknown', 'True'],
            output_dict=True
        )
        
        # Calculate macro F1 score for model selection
        val_f1_macro = val_report['macro avg']['f1-score']
        
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation macro F1: {val_f1_macro:.4f}")
        
        # Save best model based on F1 score rather than accuracy
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_model = model.state_dict().copy()
            print(f"New best model saved with F1: {val_f1_macro:.4f}")
        
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'val_f1_macro': val_f1_macro,
            'learning_rate': scheduler.get_last_lr()[0]
        })
    
    # Save the best model
    if best_model:
        torch.save(best_model, 'outputs/bert_classifier.pt')
        model.load_state_dict(best_model)
    
    return model, training_stats

def evaluate_model(model, test_dataloader, device):
    """
    Evaluate the model on the test dataset with focus on F1, precision, and recall
    """
    model.eval()
    predictions = []
    true_labels = []
    
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['False', 'Unknown', 'True'])
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'Unknown', 'True'], yticklabels=['False', 'Unknown', 'True'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('outputs/confusion_matrix.png')
    
    # Calculate and print F1, precision, recall
    report_dict = classification_report(true_labels, predictions, target_names=['False', 'Unknown', 'True'], output_dict=True)
    
    # Print focused metrics
    print("\nF1, Precision, Recall Summary:")
    print(f"Macro Avg - F1: {report_dict['macro avg']['f1-score']:.4f}, Precision: {report_dict['macro avg']['precision']:.4f}, Recall: {report_dict['macro avg']['recall']:.4f}")
    
    for cls in ['False', 'Unknown', 'True']:
        print(f"{cls} - F1: {report_dict[cls]['f1-score']:.4f}, Precision: {report_dict[cls]['precision']:.4f}, Recall: {report_dict[cls]['recall']:.4f}")
    
    return accuracy, report

def plot_training_stats(training_stats):
    """
    Plot training and validation loss/accuracy/F1
    """
    stats_df = pd.DataFrame(training_stats)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(stats_df['epoch'], stats_df['train_loss'], 'b-o', label='Training Loss')
    plt.plot(stats_df['epoch'], stats_df['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/training_loss.png')
    
    # Plot accuracy and F1
    plt.figure(figsize=(10, 6))
    plt.plot(stats_df['epoch'], stats_df['val_accuracy'], 'g-o', label='Validation Accuracy')
    plt.plot(stats_df['epoch'], stats_df['val_f1_macro'], 'm-o', label='Validation Macro F1')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/validation_metrics.png')
    
    # Plot learning rate
    if 'learning_rate' in stats_df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(stats_df['epoch'], stats_df['learning_rate'], 'b-o')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.savefig('outputs/learning_rate.png')

def predict(text, model, tokenizer, device):
    """
    Make predictions on new text
    """
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors='pt',
        max_length=128,
        padding='max_length',
        truncation=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
    confidence = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Map numerical prediction to label
    label_map = {0: 'False', 1: 'Unknown', 2: 'True'}
    predicted_label = label_map[prediction]
    
    return {
        'prediction': predicted_label,
        'confidence': {
            'False': float(confidence[0]),
            'Unknown': float(confidence[1]),
            'True': float(confidence[2])
        }
    }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load data
    train_statements, train_labels = load_data('datasets/train_set.json')
    val_statements, val_labels = load_data('datasets/validate_set.json')
    test_statements, test_labels = load_data('datasets/test_set.json')
    
    print(f"Training examples: {len(train_statements)}")
    print(f"Validation examples: {len(val_statements)}")
    print(f"Testing examples: {len(test_statements)}")
    
    train_dataset = NewsDataset(train_statements, train_labels, tokenizer)
    val_dataset = NewsDataset(val_statements, val_labels, tokenizer)
    test_dataset = NewsDataset(test_statements, test_labels, tokenizer)
    
    # Create data loaders
    batch_size = 16
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )
    
    # Train model
    print("Training model...")
    model, training_stats = train_bert_model(train_dataloader, val_dataloader, device, num_epochs=4)
    
    # Plot training stats
    plot_training_stats(training_stats)
    
    # Evaluate model
    print("\nEvaluating model on test data...")
    accuracy, report = evaluate_model(model, test_dataloader, device)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    
    # Example prediction
    example_text = "Most Americans have committed crimes worthy of prison time"
    result = predict(example_text, model, tokenizer, device)
    print(f"\nExample prediction for: '{example_text}'")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: True: {result['confidence']['True']:.4f}, False: {result['confidence']['False']:.4f}, Unknown: {result['confidence']['Unknown']:.4f}")

if __name__ == "__main__":
    main()

# Define parameter sets
parameter_sets = [
    # Set 1: Higher learning rate, lower weight decay (to combat underfitting)
    {"batch_size": 32, "learning_rate": 5e-5, "weight_decay": 0.01, "epochs": 5, 
     "use_class_weights": False, "accumulation_steps": 1},
    

    {"batch_size": 64, "learning_rate": 1e-5, "weight_decay": 0.1, "epochs": 8,
     "use_class_weights": False, "accumulation_steps": 1},
    

    {"batch_size": 32, "learning_rate": 3e-5, "weight_decay": 0.05, "epochs": 5,
     "use_class_weights": True, "accumulation_steps": 1}
]

# Load data
train_file = "train_set.json"
val_file = "validate_set.json"
test_file = "test_set.json"
train_statements, train_labels = load_data(train_file)
val_statements, val_labels = load_data(val_file)
test_statements, test_labels = load_data(test_file)

# Set up tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Combined output file
combined_output_path = "outputs/combined_evaluation_report.txt"
# Clear the combined output file if it exists
with open(combined_output_path, 'w') as f:
    f.write("Combined Evaluation Report for All Parameter Sets\n\n")

# Iterate over each parameter set
for i, params in enumerate(parameter_sets, start=1):
    print(f"\n=== Training with Parameter Set {i} ===\n")
    
    # Create a unique name for the output directory based on the set number
    output_dir = f"outputs/set{i}"
    os.makedirs(output_dir, exist_ok=True)

    # Save the parameter set to a text file
    with open(os.path.join(output_dir, 'parameters.txt'), 'w') as f:
        json.dump(params, f, indent=4)

    # Create datasets
    train_dataset = NewsDataset(train_statements, train_labels, tokenizer)
    val_dataset = NewsDataset(val_statements, val_labels, tokenizer)
    test_dataset = NewsDataset(test_statements, test_labels, tokenizer)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=params['batch_size'])
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=params['batch_size'])
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=params['batch_size'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract dropout parameters if they exist or use defaults
    hidden_dropout = params.get('hidden_dropout_prob', 0.2)
    attention_dropout = params.get('attention_probs_dropout_prob', 0.2)

    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,
        hidden_dropout_prob=hidden_dropout,
        attention_probs_dropout_prob=attention_dropout
    )
    model.to(device)

    # Set up optimizer with the baseline learning rate
    optimizer = AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    
    # Calculate total steps (number of batches * epochs)
    total_steps = len(train_dataloader) * params['epochs']
    
    # Create a scheduler with linear decay and 10% warmup
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    # Calculate class weights based on class distribution in training data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    
    # Convert to tensor for PyTorch
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Create loss function with class weights
    if params.get('use_class_weights', False):
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        loss_fct = torch.nn.CrossEntropyLoss()

    # Define accumulation steps
    accumulation_steps = params.get('accumulation_steps', 1)
    
    # Training loop with gradient accumulation and learning rate scheduling
    training_stats = []
    best_val_accuracy = 0
    best_model = None
    
    for epoch in range(params['epochs']):
        print(f"\nEpoch {epoch+1}/{params['epochs']}")
        
        # Training
        model.train()
        total_train_loss = 0
        
        # Reset gradients
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_dataloader):
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass (simpler approach)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Get loss directly from outputs if not using class weights
            if not params.get('use_class_weights', False):
                loss = outputs.loss / accumulation_steps
            else:
                # Calculate loss with class weights
                logits = outputs.logits
                loss = loss_fct(logits, labels) / accumulation_steps
            
            # Accumulate loss
            total_train_loss += loss.item() * accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights only after certain steps
            if (step + 1) % accumulation_steps == 0 or (step + 1 == len(train_dataloader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        predictions = []
        true_labels = []
        
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Calculate validation loss
            logits = outputs.logits
            loss = loss_fct(logits, labels)
            total_val_loss += loss.item()
            
            # Record predictions
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = float((torch.tensor(predictions) == torch.tensor(true_labels)).sum()) / len(true_labels)
        
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict().copy()
        
        # Record stats
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        })
    
    # Load best model
    if best_model:
        model.load_state_dict(best_model)
        torch.save(best_model, os.path.join(output_dir, 'bert_classifier_trained.pt'))
    
    # Plot training statistics
    plot_training_stats(training_stats)

    # Save the training stats
    with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
        json.dump(training_stats, f)

    # Evaluate the model
    accuracy, report = evaluate_model(model, test_dataloader, device)
    print(f"Set {i} - Test Accuracy: {accuracy:.4f}")
    print(f"Set {i} - Classification Report:\n{report}")

    # Create a comprehensive report
    report_content = {
        "parameters": params,
        "training_stats": training_stats,
        "test_accuracy": accuracy,
        "classification_report": report
    }

    # Save the comprehensive report
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(json.dumps(report_content, indent=4))

    # Append to combined output file
    with open(combined_output_path, 'a') as f:
        f.write(f"Set {i} Results:\n")
        f.write(f"Parameters: {json.dumps(params, indent=2)}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write("Final Validation Loss: {:.4f}\n".format(training_stats[-1]['val_loss']))
        f.write("Final Validation Accuracy: {:.4f}\n".format(training_stats[-1]['val_accuracy']))
        f.write("-" * 80 + "\n\n")

# After all sets are done, append summary to combined output
with open(combined_output_path, 'a') as f:
    f.write("\n\n===== SUMMARY OF ALL PARAMETER SETS =====\n\n")
    for i, params in enumerate(parameter_sets, start=1):
        report_path = os.path.join(f"outputs/set{i}", 'evaluation_report.txt')
        if os.path.exists(report_path):
            with open(report_path, 'r') as report_file:
                report_data = json.load(report_file)
                f.write(f"Set {i}: Accuracy = {report_data['test_accuracy']:.4f}, ")
                f.write(f"Parameters: {json.dumps(params)}\n")
    
    # Find best performing set
    best_set = 0
    best_accuracy = 0
    for i, params in enumerate(parameter_sets, start=1):
        report_path = os.path.join(f"outputs/set{i}", 'evaluation_report.txt')
        if os.path.exists(report_path):
            with open(report_path, 'r') as report_file:
                report_data = json.load(report_file)
                if report_data['test_accuracy'] > best_accuracy:
                    best_accuracy = report_data['test_accuracy']
                    best_set = i
    
    f.write(f"\nBEST PERFORMING SET: Set {best_set} with accuracy {best_accuracy:.4f}\n")
    f.write(f"Best parameters: {json.dumps(parameter_sets[best_set-1], indent=2)}\n")
