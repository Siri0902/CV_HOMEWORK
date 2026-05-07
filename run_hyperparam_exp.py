import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from data_loader import load_data, preprocess_data, create_dataloaders, Logger as DataLogger
from models import get_model, get_model_input_config
from trainer import Trainer


def load_exp_config(config_path='hyperparam_config.yaml'):
    """Load hyperparameter experiment configuration"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(base_dir, config_path)
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_optimizer_by_name(optimizer_name, model_parameters, learning_rate):
    """Get optimizer by name"""
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_parameters, lr=learning_rate)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=learning_rate)
    else:
        return optim.Adam(model_parameters, lr=learning_rate)


def run_single_experiment(
    model_name,
    exp_param_name,
    exp_param_value,
    default_params,
    data_dir,
    exp_output_dir,
    device,
    seed=42
):
    """Run a single hyperparameter experiment"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    exp_config = default_params.copy()
    exp_config[exp_param_name] = exp_param_value
    
    logger = DataLogger(exp_output_dir)
    logger.log("=" * 60)
    logger.log(f"实验: {exp_param_name} = {exp_param_value}")
    logger.log(f"配置: {exp_config}")
    logger.log("=" * 60)
    
    target_size, in_channels, convert_to_rgb = get_model_input_config(model_name)
    
    train_df, test_df = load_data(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        train_df, test_df,
        val_split=exp_config.get('val_split', 0.2),
        target_size=target_size,
        convert_to_rgb=convert_to_rgb
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, exp_config['batch_size']
    )
    
    model = get_model(
        model_name,
        pretrained=False,
        num_classes=exp_config.get('num_classes', 24),
        in_channels=in_channels,
        activation=exp_config.get('activation', 'ReLU')
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer_by_name(
        exp_config['optimizer'],
        model.parameters(),
        exp_config['learning_rate']
    )
    
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer,
        device, model_name, os.path.join(exp_output_dir, 'checkpoints'), logger
    )
    
    start_time = time.time()
    history, best_acc = trainer.train(exp_config['epochs'])
    train_time = time.time() - start_time
    
    result = {
        'exp_param': exp_param_name,
        'exp_value': str(exp_param_value),
        'config': exp_config,
        'best_val_acc': best_acc,
        'train_time': train_time,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }
    
    logger.save_json(result, 'result.json')
    logger.log(f"实验完成! 最佳验证准确率: {best_acc:.4f}, 训练时间: {train_time:.2f}秒")
    
    return result


def run_all_experiments(config_path='hyperparam_config.yaml'):
    """Run all hyperparameter experiments"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_exp_config(config_path)
    
    default_params = config['default']
    experiment_params = config['experiment']
    model_name = config['model']
    output_root = config.get('output_root', 'log/hyperparam')
    
    data_dir = os.path.join(base_dir, 'datasets')
    output_root_dir = os.path.join(base_dir, output_root)
    os.makedirs(output_root_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    all_results = {}
    
    for param_name, param_values in experiment_params.items():
        param_results = {}
        param_dir = os.path.join(output_root_dir, param_name)
        os.makedirs(param_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Running experiments for: {param_name}")
        print(f"Values: {param_values}")
        print(f"{'='*60}")
        
        for value in param_values:
            exp_dir_name = f"{param_name}_{value}"
            exp_output_dir = os.path.join(param_dir, exp_dir_name)
            os.makedirs(exp_output_dir, exist_ok=True)
            
            result = run_single_experiment(
                model_name=model_name,
                exp_param_name=param_name,
                exp_param_value=value,
                default_params=default_params.copy(),
                data_dir=data_dir,
                exp_output_dir=exp_output_dir,
                device=device,
                seed=config.get('seed', 42)
            )
            
            param_results[exp_dir_name] = result
        
        all_results[param_name] = param_results
    
    generate_summary(all_results, output_root_dir, default_params)
    
    return all_results


def generate_summary(all_results, output_dir, default_params):
    """Generate summary markdown and JSON files"""
    summary_md = "# 超参数实验结果汇总\n\n"
    summary_md += "## 默认参数\n\n"
    for k, v in default_params.items():
        summary_md += f"- {k}: {v}\n"
    summary_md += "\n---\n\n"
    
    summary_json = {}
    
    for param_name, param_results in all_results.items():
        summary_md += f"## {param_name}\n\n"
        summary_md += f"| 参数值 | 验证准确率 | 训练时间(秒) |\n"
        summary_md += f"|-------|----------|------------|\n"
        
        param_best = None
        param_best_acc = 0
        
        for exp_name, result in param_results.items():
            acc = result['best_val_acc']
            time_sec = result['train_time']
            value = result['exp_value']
            
            summary_md += f"| {value} | {acc:.4f} | {time_sec:.2f} |\n"
            
            summary_json[exp_name] = {
                'best_val_acc': acc,
                'train_time': time_sec
            }
            
            if acc > param_best_acc:
                param_best_acc = acc
                param_best = value
        
        summary_md += f"\n**最佳值**: {param_best} (准确率: {param_best_acc:.4f})\n\n"
        summary_json[param_name] = {
            'best_value': param_best,
            'best_acc': param_best_acc
        }
    
    summary_md += "---\n\n"
    summary_md += "## 最佳超参数推荐\n\n"
    
    best_overall = {}
    for param_name, param_results in all_results.items():
        best_acc = 0
        best_value = None
        for result in param_results.values():
            if result['best_val_acc'] > best_acc:
                best_acc = result['best_val_acc']
                best_value = result['exp_value']
        best_overall[param_name] = best_value
    
    for k, v in best_overall.items():
        summary_md += f"- {k}: {v}\n"
    
    for param_name, param_results in all_results.items():
        best_acc = 0
        best_exp = None
        for exp_name, result in param_results.items():
            if result['best_val_acc'] > best_acc:
                best_acc = result['best_val_acc']
                best_exp = exp_name
        if best_exp:
            best_overall[f'best_{param_name}_exp'] = best_exp
    
    summary_json['recommended'] = best_overall
    
    summary_file = os.path.join(output_dir, 'summary.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_md)
    
    json_file = os.path.join(output_dir, 'summary.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"\nSummary saved to:")
    print(f"  - {summary_file}")
    print(f"  - {json_file}")


if __name__ == "__main__":
    results = run_all_experiments('hyperparam_config.yaml')
    print("\nAll experiments completed!")