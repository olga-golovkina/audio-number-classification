import os
from os import path
from pathlib import Path

import dvc.api
import pandas as pd
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, annotations_path: Path, transformer, device, use_dvc=False):
        if annotations_path is None or not annotations_path.exists():
            if use_dvc:
                self.__load_from_dvc()
            else:
                raise ValueError("The dataset is not exists")

        self.dataset_dir = annotations_path.parents[0]

        self.annotations = pd.read_csv(annotations_path.resolve(), sep=";")
        self.device = device
        self.transformer = transformer

    def __len__(self):
        return len(self.annotations)

    def __get_audio_path(self, index):
        row = self.annotations.iloc[index]

        filename = row["audio"]
        dir = row["dir"]

        return path.join(self.dataset_dir, dir, filename)

    def __get_label(self, index):
        return self.annotations.iloc[index, 2]

    def __getitem__(self, index):
        audio_path = self.__get_audio_path(index)
        label = self.__get_label(index)

        signal, sample_rate = torchaudio.load(audio_path)
        signal = self.transformer.transform(signal, sample_rate)

        return signal, label

    def __load_from_dvc(self):
        dataset_folder = Path("./data")
        dataset_folder.mkdir(parents=True, exist_ok=True)

        fs = dvc.api.DVCFileSystem()

        local_train_path = Path("data/train_annotations.csv")
        local_val_path = Path("data/val_annotations.csv")

        if local_train_path.exists():
            os.remove(local_train_path.resolve())

        if local_val_path.exists():
            os.remove(local_val_path.resolve())

        fs.get("data/audio", "data/audio", recursive=True)
        fs.get_file("data/train_annotations.csv", local_train_path.resolve())
        fs.get_file("data/val_annotations.csv", local_val_path.resolve())
