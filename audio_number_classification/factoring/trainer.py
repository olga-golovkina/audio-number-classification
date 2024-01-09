from omegaconf import DictConfig
from torch import nn

from audio_number_classification.dataset import AudioDataset
from audio_number_classification.logging.logger import Logger
from audio_number_classification.trainer import Trainer


class TrainerFactory:
    @staticmethod
    def get_custom_trainer(
        model: nn.Module,
        train_dataset: AudioDataset,
        val_dataset: AudioDataset,
        logger: Logger,
        cfg: DictConfig,
    ):
        training = cfg["training"]

        return Trainer(
            model,
            train_dataset,
            val_dataset,
            logger,
            training["epochs"],
            training["batch_size"],
            training["learning_rate"],
            cfg["shared"]["device"],
        )
