import copy

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_number_classification.logging.logger import Logger

from .dataset import AudioDataset
from .model import CNNNetwork


class Trainer:
    def __init__(
        self,
        model: CNNNetwork,
        train_dataset: AudioDataset,
        val_dataset: AudioDataset,
        logger: Logger,
        epochs=4,
        batch_size=16,
        learning_rate=1e-3,
        device="cpu",
    ):
        self.model = model
        self.epochs = epochs
        self.logger = logger
        self.device = device

        self.train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size
        )
        self.valid_loader = DataLoader(
            val_dataset, shuffle=False, batch_size=batch_size
        )

        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, verbose=True)

    def fit(self):
        best_train_loss = None
        best_val_loss = None
        best_model = None

        for _ in range(1, self.epochs + 1):
            train_loss, train_acc = self.__train_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
            )
            valid_loss, valid_acc = self.__valid_epoch(
                self.model, self.valid_loader, self.criterion, self.device
            )

            self.scheduler.step()

            if self.__is_loss_valid(
                train_loss, best_train_loss, valid_loss, best_val_loss
            ):
                best_train_loss = train_loss
                best_val_loss = valid_loss
                best_model = copy.deepcopy(self.model)

            self.logger.log_metrics(
                {
                    "train loss": train_loss,
                    "train accuracy": train_acc,
                    "valid loss": valid_loss,
                    "valid accuracy": valid_acc,
                }
            )

        self.logger.close()

        return best_model, best_train_loss, best_val_loss

    def __train_epoch(self, model, train_loader, criterion, optimizer, device):
        model.train()

        train_loss = 0.0

        target_cls = []
        pred_cls = []

        for props, target in tqdm(train_loader, desc="Train: "):
            optimizer.zero_grad()

            props, target = props.to(device), target.to(device)

            pred = model(props)

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            target_cls.extend(target.squeeze().to(torch.long))
            pred_cls.extend(torch.round(pred.squeeze()).to(torch.long))

        train_loss /= len(train_loader)
        train_acc = float(
            np.mean(np.array(target_cls) == np.array([p.max() for p in pred_cls]))
        )

        return train_loss, train_acc

    def __valid_epoch(self, model, valid_loader, criterion, device):
        model.eval()

        valid_loss = 0.0

        target_cls = []
        pred_cls = []

        with torch.no_grad():
            for props, target in valid_loader:
                props, target = props.to(device), target.to(device)

                pred = model(props)

                loss = criterion(pred, target)
                valid_loss += loss.item()

                target_cls.extend(target.squeeze().to(torch.long))
                pred_cls.extend(torch.round(pred.squeeze()).to(torch.long))

        valid_loss /= len(valid_loader)
        valid_acc = float(
            np.mean(np.array(target_cls) == np.array([p.max() for p in pred_cls]))
        )

        return valid_loss, valid_acc

    def __is_loss_valid(self, train_loss, best_train_loss, val_loss, best_val_loss):
        if best_train_loss is None or best_val_loss is None:
            return True

        best_diff_loss = abs(best_train_loss - best_val_loss)
        diff_loss = abs(train_loss - val_loss)

        return (
            best_train_loss >= train_loss
            and best_val_loss >= val_loss
            and best_diff_loss >= diff_loss
        )
