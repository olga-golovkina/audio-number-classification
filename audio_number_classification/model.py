import os
import warnings
from pathlib import Path

import dvc.api
import torch
from torch import nn


class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        conv_params = {
            "in_channels": 1,
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 2,
        }

        self.conv1 = nn.Sequential(
            nn.Conv2d(**conv_params),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        conv_params["in_channels"] = 16
        conv_params["out_channels"] = 32

        self.conv2 = nn.Sequential(
            nn.Conv2d(**conv_params),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        conv_params["in_channels"] = 32
        conv_params["out_channels"] = 64

        self.conv3 = nn.Sequential(
            nn.Conv2d(**conv_params),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        conv_params["in_channels"] = 64
        conv_params["out_channels"] = 128

        self.conv4 = nn.Sequential(
            nn.Conv2d(**conv_params),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 10)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)
        logits = self.linear(x)

        return logits

    def save(self, model_path: Path, use_dvc=False):
        if model_path is None:
            raise TypeError("model_path must by Path type")

        if not model_path.is_file():
            raise ValueError("model_path must be file path")

        model_path.parents[0].mkdir(parents=True, exist_ok=True)

        save_path = model_path.resolve()
        torch.save(self.state_dict(), save_path)

        if use_dvc:
            os.system(f"dvc add {save_path}")
            os.system("dvc commit -m 'Add model'")
            os.system("dvc push")

    def load(self, model_path: Path = None, use_dvc: bool = False):
        fs = dvc.api.DVCFileSystem()

        if model_path.is_file() and model_path.exists():
            warnings.warn(
                "The model is already downloaded. "
                "If you want to update it you should to delete old model.",
                stacklevel=2,
            )
            return

        if not fs.exists("./models/model.pt"):
            warnings.warn("The model is not exists in remote storage", stacklevel=2)
            return

        fs.get_file("./models/model.pt", model_path)
