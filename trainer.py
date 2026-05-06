import os

import torch


class Trainer:
    """训练器类"""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device, model_name, checkpoint_dir, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        self.history = None
        self.best_val_acc = 0.0

    def train(self, num_epochs):
        self.model = self.model.to(self.device)
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        best_val_acc = 0.0
        best_model_state = None

        self.logger.log(f"开始训练 {self.model_name}...")

        for epoch in range(num_epochs):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.logger.log(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                )

        self.best_val_acc = best_val_acc
        if best_model_state:
            self._save_checkpoint(best_model_state)

        self.logger.log(
            f"{self.model_name} 训练完成! 最佳验证准确率: {best_val_acc:.4f}"
        )

        return self.history, best_val_acc

    def _train_epoch(self):
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        return train_loss, train_acc

    def _validate(self):
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        return val_loss, val_acc

    def _save_checkpoint(self, state):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(self.checkpoint_dir, f'{self.model_name}.pth')
        torch.save(state, filepath)
        self.logger.log(f"模型已保存: {filepath}")


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return torch.tensor(all_labels), torch.tensor(all_preds)


def load_model(model_class, checkpoint_path, device, num_classes=24, **kwargs):
    """加载模型"""
    model = model_class(num_classes=num_classes, **kwargs)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model
