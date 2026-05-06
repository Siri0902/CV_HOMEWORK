import os

import torch
import yaml


class Config:
    """配置加载类"""

    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self):
        """加载YAML配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    @property
    def pretrained(self):
        return self._config.get('model', {}).get('pretrained', False)

    @property
    def num_epochs(self):
        return self._config.get('training', {}).get('num_epochs', 20)

    @property
    def batch_size(self):
        return self._config.get('training', {}).get('batch_size', 64)

    @property
    def learning_rate(self):
        return self._config.get('training', {}).get('learning_rate', 0.001)

    @property
    def val_split(self):
        return self._config.get('training', {}).get('val_split', 0.2)

    @property
    def num_classes(self):
        return self._config.get('data', {}).get('num_classes', 24)

    @property
    def in_channels(self):
        return self._config.get('data', {}).get('in_channels', 1)

    @property
    def image_size(self):
        return self._config.get('data', {}).get('image_size', 28)

    @property
    def models(self):
        return self._config.get('models', ['MyDNN'])

    @property
    def seed(self):
        return self._config.get('seed', 42)

    @property
    def save_checkpoint(self):
        return self._config.get('save_checkpoint', True)

    @property
    def verbose(self):
        return self._config.get('verbose', True)

    @property
    def data_dir(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, 'datasets')

    @property
    def log_dir(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, 'log')

    @property
    def checkpoint_dir(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, 'checkpoints')

    def set_seed(self):
        """设置随机种子"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

    def __getitem__(self, key):
        return self._config.get(key)

    def __repr__(self):
        return f"Config(num_epochs={self.num_epochs}, batch_size={self.batch_size}, models={self.models})"


if __name__ == "__main__":
    config = Config()
    print(config)
