import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report

from config import Config
from data_loader import load_data, preprocess_data, create_dataloaders, get_label_mapping, Logger
from models import get_model, get_model_input_config
from trainer import evaluate_model


def plot_training_curves(results, log_dir):
    plots_dir = os.path.join(log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cmap = plt.colormaps['tab10']

    for idx, (name, res) in enumerate(results.items()):
        history = res['history']
        color = cmap(idx % 10)
        axes[0, 0].plot(history['train_loss'], label=f'{name}', color=color)
        axes[0, 1].plot(history['train_acc'], label=f'{name}', color=color)

    for idx, (name, res) in enumerate(results.items()):
        history = res['history']
        color = cmap(idx % 10)
        axes[1, 0].plot(history['val_loss'], label=f'{name}', color=color)
        axes[1, 1].plot(history['val_acc'], label=f'{name}', color=color)

    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].set_title('Validation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print("Training curves saved")


def evaluate_best_model(config, device):
    log_dir = config.log_dir
    checkpoint_dir = config.checkpoint_dir

    with open(os.path.join(log_dir, 'results.json'), 'r') as f:
        results = json.load(f)

    logger = Logger(log_dir)
    best_model_name = max(results, key=lambda x: results[x]['best_val_acc'])
    logger.log(f"评估最佳模型: {best_model_name}")

    model = get_model(
        best_model_name,
        pretrained=False,
        num_classes=config.num_classes,
        in_channels=config.in_channels
    )

    checkpoint_path = os.path.join(checkpoint_dir, f'{best_model_name}.pth')
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    target_size, _, _ = get_model_input_config(best_model_name)
    train_df, test_df = load_data(config.data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        train_df, test_df, config.val_split, target_size=target_size
    )
    test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, config.batch_size
    )[2]

    y_true, y_pred = evaluate_model(model, test_loader, device)
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    letters = list(get_label_mapping().values())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=letters, yticklabels=letters)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plots_dir = os.path.join(log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    report = classification_report(y_true, y_pred, target_names=letters, output_dict=True)
    with open(os.path.join(log_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    accuracy = (y_true == y_pred).mean()
    logger.log(f"Test Accuracy: {accuracy:.4f}")

    eval_results = {
        'model': best_model_name,
        'test_accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    logger.save_json(eval_results, 'evaluation_results.json')

    logger.log("Evaluation completed!")
    return eval_results, y_true, y_pred


def plot_sample_predictions(X_test, y_true, y_pred, log_dir):
    letters = list(get_label_mapping().values())
    plots_dir = os.path.join(log_dir, 'plots')

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()

    for i in range(10):
        true_label = y_true[i]
        pred_label = y_pred[i]
        axes[i].imshow(X_test[i].squeeze(), cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {letters[true_label]}\nPred: {letters[pred_label]}', color=color)
        axes[i].axis('off')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)')
    plt.subplots_adjust(hspace=0.5, top=0.88)
    plt.savefig(os.path.join(plots_dir, 'sample_predictions.png'), dpi=150)
    plt.close()
    print("Sample predictions saved")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = Config(os.path.join(base_dir, 'config.yaml'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(config.log_dir, 'results.json'), 'r') as f:
        results = json.load(f)

    best_model_name = max(results.keys(), key=lambda k: results[k].get('val_accuracy', 0))
    target_size, _, _ = get_model_input_config(best_model_name)

    plot_training_curves(results, config.log_dir)

    train_df, test_df = load_data(config.data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        train_df, test_df, config.val_split, target_size=target_size
    )

    eval_results, y_true, y_pred = evaluate_best_model(config, device)

    plot_sample_predictions(X_test, y_true, y_pred, config.log_dir)

    print("Evaluation completed!")


if __name__ == "__main__":
    main()
