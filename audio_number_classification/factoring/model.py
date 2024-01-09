from pathlib import Path

from omegaconf import DictConfig

from audio_number_classification.model import CNNNetwork


class ModelFactory:
    @staticmethod
    def create_empty_model(cfg: DictConfig):
        device = cfg["shared"]["device"]

        return CNNNetwork().to(device)

    @staticmethod
    def get_model_from(cfg: DictConfig):
        model = ModelFactory.create_empty_model(cfg)

        model.load(Path(cfg["path"]["model"]), cfg["model"]["pull_dvc"])

        return model
