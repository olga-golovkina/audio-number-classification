from pathlib import Path

from omegaconf import DictConfig

from audio_number_classification.augmentation import AudioTransformer
from audio_number_classification.dataset import AudioDataset


class DatasetFactory:
    @staticmethod
    def get_dataset(trans: AudioTransformer, cfg: DictConfig, annotations_path: Path):
        device = cfg["shared"]["device"]

        return AudioDataset(
            annotations_path,
            trans,
            device,
            cfg["dataset"]["pull_dvc"],
        )
