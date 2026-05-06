import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_label_mapping():
    """获取标签到字母的映射"""
    letters = {i: chr(ord('A') + i) for i in range(25)}
    letters.pop(9)  # 移除J
    return letters


class Logger:
    """日志记录类"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'training.log')

    def log(self, message):
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def save_json(self, data, filename):
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_markdown(self, content, filename):
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)


def load_data(data_dir):
    """加载CSV数据"""
    train_csv = os.path.join(data_dir, 'sign_mnist_train.csv')
    test_csv = os.path.join(data_dir, 'sign_mnist_test.csv')

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    return train_df, test_df


def preprocess_data(train_df, test_df, val_split=0.2):
    """数据预处理"""
    y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values

    y_test = test_df['label'].values
    X_test = test_df.drop('label', axis=1).values

    X_train = X_train.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=42, stratify=y_train
    )

    return X_train_split, X_val, X_test, y_train_split, y_val, y_test


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64):
    """创建DataLoader"""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from config import Config
    config = Config()
    train_df, test_df = load_data(config.data_dir)
    print(f"训练数据: {train_df.shape}")
    print(f"测试数据: {test_df.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(train_df, test_df, config.val_split)
    print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")