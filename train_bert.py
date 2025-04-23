import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from bert_classification import load_data, NewsDataset, train_bert_model, evaluate_model, plot_training_stats

# Define parameter sets
parameter_sets = [
    {"batch_size": 16, "learning_rate": 2e-5, "weight_decay": 0.01, "epochs": 4, "dropout": 0.1},
]

def visualize_metrics(test_labels, test_preds, training_stats, output_dir):
    """
    Create comprehensive visualizations for model performance metrics.
    
    Args:
        test_labels: True labels from test set
        test_preds: Predicted labels from test set
        training_stats: List of dictionaries containing training statistics per epoch
        output_dir: Directory to save visualizations
    """
    # Create directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    stats_df = pd.DataFrame(training_stats)
    
    # 1. Training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(stats_df['epoch'], stats_df['train_loss'], 'b-o', linewidth=2, label='Training Loss')
    plt.plot(stats_df['epoch'], stats_df['val_loss'], 'r-o', linewidth=2, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'loss_curves.png'), dpi=200)
    plt.close()
    
    # 2. Validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(stats_df['epoch'], stats_df['val_accuracy'], 'g-o', linewidth=2, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'accuracy_curve.png'), dpi=200)
    plt.close()
    
    # 3. Precision, Recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, labels=[0, 1, 2], average=None
    )
    
    class_metrics = pd.DataFrame({
        'Class': ['False', 'Unknown', 'True'],
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Bar chart
    plt.figure(figsize=(12, 7))
    metrics_df = pd.melt(class_metrics, 
                       id_vars=['Class'], 
                       value_vars=['Precision', 'Recall', 'F1-Score'],
                       var_name='Metric', value_name='Value')
    
    # Plot bar chart
    sns.barplot(x='Class', y='Value', hue='Metric', data=metrics_df, palette='viridis')
    plt.title('Precision, Recall, and F1-Score by Class')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'class_metrics.png'), dpi=200)
    plt.close()
    
    # 4. Enhanced confusion matrix
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1, 2])
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot = np.empty_like(cm, dtype=str)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
              xticklabels=['False', 'Unknown', 'True'], 
              yticklabels=['False', 'Unknown', 'True'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'), dpi=200)
    plt.close()
    
    # 5. Calculate and visualize macro and weighted averages
    macro_avg = np.mean([precision, recall, f1], axis=1)
    weighted_avg = np.average([precision, recall, f1], weights=support, axis=1)
    
    avg_metrics = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Macro Average': macro_avg,
        'Weighted Average': weighted_avg
    })
    
    plt.figure(figsize=(10, 6))
    avg_df = pd.melt(avg_metrics, 
                    id_vars=['Metric'], 
                    value_vars=['Macro Average', 'Weighted Average'],
                    var_name='Average Type', value_name='Value')
    
    sns.barplot(x='Metric', y='Value', hue='Average Type', data=avg_df, palette='Set2')
    plt.title('Macro and Weighted Average Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'average_metrics.png'), dpi=200)
    plt.close()
    
    class_metrics.to_csv(os.path.join(vis_dir, 'class_metrics.csv'), index=False)
    
    print(f"\nVisualization complete! The following charts have been saved to {vis_dir}:")
    print(f"- Loss curves: loss_curves.png")
    print(f"- Accuracy curve: accuracy_curve.png")
    print(f"- Class metrics (precision, recall, F1): class_metrics.png")
    print(f"- Confusion matrix: confusion_matrix.png")
    print(f"- Average metrics: average_metrics.png")

# Load data
train_file = "train_set.json"
val_file = "validate_set.json"
test_file = "test_set.json"
train_statements, train_labels = load_data(train_file)
val_statements, val_labels = load_data(val_file)
test_statements, test_labels = load_data(test_file)

# Set up tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

combined_output_path = "outputs/combined_evaluation_report.txt"

# Iterate over each parameter set
for i, params in enumerate(parameter_sets, start=1):
    # Create a unique name for the output directory based on the set number
    output_dir = f"outputs/set{i}"
    os.makedirs(output_dir, exist_ok=True)

    # Save the parameter set to a text file
    with open(os.path.join(output_dir, 'parameters.txt'), 'w') as f:
        json.dump(params, f, indent=4)

    # Create datasets
    train_dataset = NewsDataset(train_statements, train_labels, tokenizer)
    val_dataset = val_dataset = NewsDataset(val_statements, val_labels, tokenizer)
    test_dataset = test_dataset = NewsDataset(test_statements, test_labels, tokenizer)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=params['batch_size'])
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=params['batch_size'])
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=params['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3
    )
    model.to(device)

    # Set up optimizer with the baseline learning rate
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    
    # Calculate total steps (number of batches * epochs)
    total_steps = len(train_dataloader) * 8
    
    # Create a scheduler with linear decay and 10% warmup
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    # Train the model
    model, training_stats = train_bert_model(train_dataloader, val_dataloader, device, num_epochs=params['epochs'])

    # Plot training statistics
    plot_training_stats(training_stats)

    # Save the model and training stats
    torch.save(model.state_dict(), os.path.join(output_dir, 'bert_classifier_trained.pt'))
    with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
        json.dump(training_stats, f)

    # Evaluate the model
    accuracy, report = evaluate_model(model, test_dataloader, device)
    print(f"Set {i} - Test Accuracy: {accuracy:.4f}")
    print(f"Set {i} - Classification Report:\n{report}")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    visualize_metrics(all_labels, all_preds, training_stats, output_dir)

    report_content = {
        "parameters": params,
        "training_stats": training_stats,
        "test_accuracy": accuracy,
        "classification_report": report
    }

    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(json.dumps(report_content, indent=4))

    with open(combined_output_path, 'a') as f:
        f.write(f"Set {i} Results:\n")
        f.write(json.dumps(report_content, indent=4))
        f.write("\n\n")