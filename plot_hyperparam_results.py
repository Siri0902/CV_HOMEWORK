import json
import os
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = 'log/hyperparam'
PLOTS_DIR = os.path.join(LOG_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

with open(os.path.join(LOG_DIR, 'summary.json'), 'r') as f:
    summary = json.load(f)

def plot_hyperparam_comparison():
    """Plot hyperparameter comparison charts"""
    params = ['batch_size', 'epochs', 'learning_rate', 'optimizer', 'activation']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    param_configs = {
        'batch_size': ['16', '32', '64', '128', '256'],
        'epochs': ['5', '10', '20', '30'],
        'learning_rate': ['0.0001', '0.0005', '0.001', '0.005', '0.01'],
        'optimizer': ['Adam', 'SGD', 'RMSprop'],
        'activation': ['ReLU', 'GELU', 'ELU']
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, param in enumerate(params):
        ax = axes[idx]
        configs = param_configs[param]
        
        accs = []
        times = []
        for cfg in configs:
            exp_name = f"{param}_{cfg}"
            if param == 'learning_rate':
                exp_name = f"{param}_{cfg}"
            result = summary.get(exp_name, {})
            accs.append(result.get('best_val_acc', 0) * 100)
            times.append(result.get('train_time', 0))
        
        x = np.arange(len(configs))
        width = 0.35
        
        ax.bar(x - width/2, accs, width, label='Accuracy (%)', color='steelblue', alpha=0.8)
        ax.set_xlabel(param)
        ax.set_ylabel('Accuracy (%)', color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45)
        ax.set_ylim(95, 101)
        ax.set_title(f'{param} vs Accuracy')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.bar(x + width/2, times, width, label='Time (s)', color='coral', alpha=0.8)
        ax2.set_ylabel('Training Time (s)', color='coral')
        
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'hyperparam_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/hyperparam_comparison.png")


def plot_training_curves():
    """Plot training curves for each hyperparameter"""
    param_dirs = ['batch_size', 'epochs', 'learning_rate', 'optimizer', 'activation']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, param_dir in enumerate(param_dirs):
        ax = axes[idx]
        
        param_dir_path = os.path.join(LOG_DIR, param_dir)
        if not os.path.isdir(param_dir_path):
            continue
        
        for subdir in os.listdir(param_dir_path):
            result_file = os.path.join(param_dir_path, subdir, 'result.json')
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    result = json.load(f)
                
                history = result.get('history', {})
                if history:
                    val_acc = history.get('val_acc', [])
                    if val_acc:
                        ax.plot(val_acc, label=subdir.replace('batch_size_', '').replace('epochs_', '').replace('learning_rate_', '').replace('optimizer_', '').replace('activation_', ''))
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title(f'{param_dir} - Training Curves')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/training_curves.png")


def plot_loss_curves():
    """Plot loss curves for selected experiments"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    experiments = [
        ('batch_size', 'batch_size_128'),
        ('epochs', 'epochs_20'),
        ('learning_rate', 'learning_rate_0.001'),
        ('optimizer', 'optimizer_Adam'),
        ('activation', 'activation_ReLU')
    ]
    
    for idx, (param, exp_name) in enumerate(experiments):
        ax = axes[idx]
        
        result_file = os.path.join(LOG_DIR, param, exp_name, 'result.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            history = result.get('history', {})
            if history:
                train_loss = history.get('train_loss', [])
                val_loss = history.get('val_loss', [])
                epochs = range(1, len(train_loss) + 1)
                
                ax.plot(epochs, train_loss, label='Train Loss', color='steelblue')
                ax.plot(epochs, val_loss, label='Val Loss', color='coral')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'{exp_name} - Loss Curves')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/loss_curves.png")


def plot_time_comparison():
    """Plot training time comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    params = ['batch_size', 'epochs', 'learning_rate', 'optimizer', 'activation']
    param_configs = {
        'batch_size': ['16', '32', '64', '128', '256'],
        'epochs': ['5', '10', '20', '30'],
        'learning_rate': ['0.0001', '0.0005', '0.001', '0.005', '0.01'],
        'optimizer': ['Adam', 'SGD', 'RMSprop'],
        'activation': ['ReLU', 'GELU', 'ELU']
    }
    
    all_times = []
    all_labels = []
    colors = []
    
    color_map = {'batch_size': '#1f77b4', 'epochs': '#ff7f0e', 'learning_rate': '#2ca02c', 
                'optimizer': '#d62728', 'activation': '#9467bd'}
    
    for param in params:
        configs = param_configs[param]
        for cfg in configs:
            exp_name = f"{param}_{cfg}"
            result = summary.get(exp_name, {})
            all_times.append(result.get('train_time', 0))
            all_labels.append(f"{param}={cfg}")
            colors.append(color_map[param])
    
    x = np.arange(len(all_times))
    ax.bar(x, all_times, color=colors, alpha=0.8)
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison Across All Hyperparameter Experiments')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=90, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'time_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/time_comparison.png")


def create_summary_table():
    """Create summary table image"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    table_data = [
        ['Batch Size', '16', '100.00%', '184.86s'],
        ['', '32', '100.00%', '93.09s'],
        ['', '64', '100.00%', '48.62s'],
        ['', '128', '100.00%', '27.15s'],
        ['', '256', '100.00%', '19.25s'],
        ['Epochs', '5', '100.00%', '6.70s'],
        ['', '10', '100.00%', '13.26s'],
        ['', '20', '100.00%', '26.31s'],
        ['', '30', '100.00%', '39.24s'],
        ['Learning Rate', '0.0001', '100.00%', '25.85s'],
        ['', '0.001', '100.00%', '26.20s'],
        ['', '0.01', '100.00%', '27.03s'],
        ['Optimizer', 'Adam', '100.00%', '26.68s'],
        ['', 'SGD', '100.00%', '25.97s'],
        ['', 'RMSprop', '100.00%', '26.72s'],
        ['Activation', 'ReLU', '100.00%', '27.71s'],
        ['', 'GELU', '100.00%', '26.77s'],
        ['', 'ELU', '100.00%', '26.36s'],
    ]
    
    table = ax.table(cellText=table_data,
                   colLabels=['Hyperparameter', 'Value', 'Val Accuracy', 'Time'],
                   loc='center',
                   cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    ax.set_title('Hyperparameter Experiment Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/summary_table.png")


if __name__ == "__main__":
    print("Generating hyperparameter experiment plots...")
    plot_hyperparam_comparison()
    plot_training_curves()
    plot_loss_curves()
    plot_time_comparison()
    create_summary_table()
    print("All plots generated successfully!")