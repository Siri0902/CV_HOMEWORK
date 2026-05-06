import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from data_loader import load_data, preprocess_data, create_dataloaders, Logger
from models import get_model
from trainer import Trainer


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = Config(os.path.join(base_dir, 'config.yaml'))
    config.set_seed()

    logger = Logger(config.log_dir)
    logger.log("=" * 60)
    logger.log("手语识别模型训练开始")
    logger.log("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"使用设备: {device}")

    logger.log("加载数据...")
    train_df, test_df = load_data(config.data_dir)
    logger.log(f"训练数据: {train_df.shape}, 测试数据: {test_df.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        train_df, test_df, config.val_split
    )
    logger.log(
        f"数据划分: 训练集={X_train.shape[0]}, "
        f"验证集={X_val.shape[0]}, 测试集={X_test.shape[0]}"
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, config.batch_size
    )

    model_names = config.models
    results = {}

    for model_name in model_names:
        logger.log("-" * 40)
        logger.log(f"训练模型: {model_name}")
        logger.log("-" * 40)

        model = get_model(
            model_name,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            in_channels=config.in_channels
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        trainer = Trainer(
            model, train_loader, val_loader, criterion, optimizer,
            device, model_name, config.checkpoint_dir, logger
        )

        start_time = time.time()
        history, best_acc = trainer.train(config.num_epochs)
        train_time = time.time() - start_time

        results[model_name] = {
            'history': history,
            'best_val_acc': best_acc,
            'train_time': train_time
        }

        logger.log(f"{model_name} 完成! 训练时间: {train_time:.2f}秒")

    logger.log("=" * 60)
    logger.log("所有模型训练完成!")
    logger.log("=" * 60)

    comparison_md = "# 模型性能比较\n\n"
    comparison_md += "| 模型 | 最佳验证准确率 | 训练时间(秒) |\n"
    comparison_md += "|------|------------|------------|\n"
    for name, res in results.items():
        comparison_md += f"| {name} | {res['best_val_acc']:.4f} | {res['train_time']:.2f} |\n"

    logger.save_markdown(comparison_md, 'model_comparison.md')
    logger.log("\n" + comparison_md)

    results_json = {
        name: {
            'best_val_acc': float(res['best_val_acc']),
            'train_time': float(res['train_time']),
            'history': {k: [float(v) for v in vals] for k, vals in res['history'].items()}
        }
        for name, res in results.items()
    }
    logger.save_json(results_json, 'results.json')
    logger.log("结果已保存到 results.json")

    best_model_name = max(results, key=lambda x: results[x]['best_val_acc'])
    logger.log(f"最佳模型: {best_model_name} (准确率: {results[best_model_name]['best_val_acc']:.4f})")

    logger.log("训练完成!")


if __name__ == "__main__":
    main()
