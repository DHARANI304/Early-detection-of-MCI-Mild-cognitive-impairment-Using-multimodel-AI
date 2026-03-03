import os
import torch
import numpy as np
from mri_model import AlzheimerMRIClassifier, MRIConfig, train_new_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def plot_confusion_matrix(cm, classes, title):
    """Plot confusion matrix with a heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_training_history(history, model_name):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower()}_training_history.png')
    plt.close()

def evaluate_model(model, test_loader, device, model_name):
    """Evaluate model and generate metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Get class names
    class_names = list(MRIConfig.CLASS_NAMES.values())
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, f'{model_name} Confusion Matrix')
    
    # Print classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=class_names,
                                 output_dict=True)
    
    return cm, report

def main():
    print("Starting Alzheimer's MRI Model Training and Evaluation")
    print("-" * 50)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Dictionary to store results
    results = {
        'vgg': {},
        'densenet': {}
    }
    
    # Train and evaluate VGG16
    print("\nTraining VGG16 Model...")
    vgg_classifier = AlzheimerMRIClassifier("vgg")
    train_loader, test_loader = vgg_classifier.load_data()
    vgg_model, vgg_history = vgg_classifier.train_model(return_history=True)
    
    print("\nEvaluating VGG16 Model...")
    vgg_cm, vgg_report = evaluate_model(
        vgg_classifier.model, 
        test_loader, 
        MRIConfig.DEVICE,
        "VGG16"
    )
    
    # Plot VGG16 training history
    plot_training_history(vgg_history, "VGG16")
    results['vgg'] = {
        'accuracy': vgg_report['accuracy'],
        'per_class': vgg_report,
        'confusion_matrix': vgg_cm
    }
    
    # Train and evaluate DenseNet
    print("\nTraining DenseNet Model...")
    densenet_classifier = AlzheimerMRIClassifier("densenet")
    train_loader, test_loader = densenet_classifier.load_data()
    densenet_model, densenet_history = densenet_classifier.train_model(return_history=True)
    
    print("\nEvaluating DenseNet Model...")
    densenet_cm, densenet_report = evaluate_model(
        densenet_classifier.model,
        test_loader,
        MRIConfig.DEVICE,
        "DenseNet"
    )
    
    # Plot DenseNet training history
    plot_training_history(densenet_history, "DenseNet")
    results['densenet'] = {
        'accuracy': densenet_report['accuracy'],
        'per_class': densenet_report,
        'confusion_matrix': densenet_cm
    }
    
    # Print comparative results
    print("\nModel Comparison Results:")
    print("-" * 50)
    print(f"VGG16 Accuracy: {results['vgg']['accuracy']:.4f}")
    print(f"DenseNet Accuracy: {results['densenet']['accuracy']:.4f}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'results/model_comparison_{timestamp}.txt', 'w') as f:
        f.write("Alzheimer's MRI Classification Model Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, model_results in results.items():
            f.write(f"\n{model_name.upper()} Results:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Accuracy: {model_results['accuracy']:.4f}\n\n")
            
            f.write("Per-Class Performance:\n")
            for class_name, metrics in model_results['per_class'].items():
                if isinstance(metrics, dict):
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
    
    print(f"\nResults saved to results/model_comparison_{timestamp}.txt")
    print("Training histories and confusion matrices saved in the results/ directory")

if __name__ == "__main__":
    main()