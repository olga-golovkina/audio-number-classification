from pathlib import Path

import torch
import tqdm
from dataset import AudioDataset
from logger import Logger
from model import CNNNetwork
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: CNNNetwork,
        train_dataset: AudioDataset,
        val_dataset: AudioDataset,
        logger: Logger,
        epochs=4,
        batch_size=16,
        device="cpu",
        model_save_name: Path = None,
    ):
        self.model = model
        self.model_save_name = model_save_name
        self.epochs = epochs
        self.logger = logger
        self.device = device

        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size
        )
        self.val_dataloader = DataLoader(
            val_dataset, shuffle=False, batch_size=batch_size
        )

        self.optimizer = Adam(model.parameters(), lr=1e-3)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, verbose=True)

    def fit(self):
        train_losses = []
        valid_losses = []

        best_train_loss = None
        best_val_loss = None
        best_model_path = None

        for _ in range(1, self.epochs + 1):
            train_loss = self.__train_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
            )
            valid_loss = self.__valid_epoch(
                self.model, self.valid_loader, self.criterion, self.device
            )

            self.scheduler.step()

            if self.__is_loss_valid(
                train_loss, best_train_loss, valid_loss, best_val_loss
            ):
                best_train_loss = train_loss
                best_val_loss = valid_loss
                self.model.save(self.model_save_name)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print(f"Current train loss: {train_loss}")
            print(f"Current validation loss: {valid_loss}")
            print(f"Current loss difference: {abs(train_loss - valid_loss)}")

        return best_model_path, best_train_loss, best_val_loss

    def __train_epoch(self, model, train_loader, criterion, optimizer, device):
        model.train()

        train_loss = 0.0

        for props, target in tqdm(train_loader, desc="Train: "):
            optimizer.zero_grad()

            props, target = props.to(device), target.to(device)

            pred = model(props)

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        return train_loss

    def __valid_epoch(self, model, valid_loader, criterion, device):
        model.eval()

        valid_loss = 0.0
        with torch.no_grad():
            for props, target in valid_loader:
                props, target = props.to(device), target.to(device)

                pred = model(props)

                loss = criterion(pred, target)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)

        return valid_loss

    def __is_loss_valid(train_loss, best_train_loss, val_loss, best_val_loss):
        if best_train_loss is None or best_val_loss is None:
            return True

        best_diff_loss = abs(best_train_loss - best_val_loss)
        diff_loss = abs(train_loss - val_loss)

        return (
            best_train_loss >= train_loss
            and best_val_loss >= val_loss
            and best_diff_loss >= diff_loss
        )
