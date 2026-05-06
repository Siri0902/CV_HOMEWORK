import cv2
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def get_label_mapping():
    """获取标签到字母的映射 (24类, A-Y不含J)"""
    letters = []
    for i in range(25):
        if i != 9:  # 跳过J
            letters.append(chr(ord('A') + i))
    return {i: letter for i, letter in enumerate(letters)}


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


def preprocess_data(train_df, test_df, val_split=0.2, target_size=224, convert_to_rgb=False):
    """数据预处理
    
    Args:
        train_df: 训练数据DataFrame
        test_df: 测试数据DataFrame
        val_split: 验证集比例
        target_size: 目标图像尺寸，默认224
        convert_to_rgb: 是否转换为3通道（适配AlexNet/VGG/ResNet等标准模型）
    """
    y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values

    y_test = test_df['label'].values
    X_test = test_df.drop('label', axis=1).values

    # 标签映射：将原始标签(0-24但跳过9)映射到连续索引(0-23)
    # 原始: A=0, B=1, ..., I=8, J=9, K=10, ..., Y=24
    # 目标: A=0, B=1, ..., I=8, K=9, ..., Y=23 (跳过J)
    label_map = {}
    original_labels = sorted(set(y_train) | set(y_test))
    for idx, orig_label in enumerate(original_labels):
        label_map[orig_label] = idx

    y_train = np.array([label_map[l] for l in y_train])
    y_test = np.array([label_map[l] for l in y_test])

    X_train = X_train.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # 图像缩放：28x28 -> target_size x target_size
    if target_size != 28:
        X_train_scaled = np.zeros((X_train.shape[0], target_size, target_size), dtype=np.float32)
        X_test_scaled = np.zeros((X_test.shape[0], target_size, target_size), dtype=np.float32)
        
        for i in range(X_train.shape[0]):
            img_28 = (X_train[i] * 255).astype(np.uint8)
            img_scaled = cv2.resize(img_28, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            X_train_scaled[i] = img_scaled.astype(np.float32) / 255.0
        
        for i in range(X_test.shape[0]):
            img_28 = (X_test[i] * 255).astype(np.uint8)
            img_scaled = cv2.resize(img_28, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            X_test_scaled[i] = img_scaled.astype(np.float32) / 255.0
        
        X_train = X_train_scaled
        X_test = X_test_scaled

    # 转换为3通道（如需适配标准模型）
    if convert_to_rgb:
        X_train = np.stack([X_train] * 3, axis=1)
        X_test = np.stack([X_test] * 3, axis=1)
    else:
        X_train = X_train[:, np.newaxis, :, :]
        X_test = X_test[:, np.newaxis, :, :]

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=42, stratify=y_train
    )

    return X_train_split, X_val, X_test, y_train_split, y_val, y_test


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64):
    """创建DataLoader"""
    # 复制数组以避免PyTorch警告
    X_train = np.copy(X_train)
    X_val = np.copy(X_val)
    X_test = np.copy(X_test)
    y_train = np.copy(y_train)
    y_val = np.copy(y_val)
    y_test = np.copy(y_test)

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

    # 根据模型类型自动选择预处理参数
    model_type = "resnet"  # 示例：可切换为 'cnn' 或 'resnet' 等
    target_size = 224 if model_type in ["alexnet", "vgg", "resnet"] else 28
    convert_to_rgb = model_type in ["alexnet", "vgg", "resnet"]

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        train_df, test_df, 
        val_split=config.val_split,
        target_size=target_size,
        convert_to_rgb=convert_to_rgb
    )
    print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
    print(f"输入张量形状 (B, C, H, W): {X_train.shape}")
