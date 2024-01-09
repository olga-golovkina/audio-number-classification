from omegaconf import DictConfig

from audio_number_classification.logging.mlflow_logger import MLFlowLogger
from audio_number_classification.logging.wandb_logger import WandbLoggerDecorator


class LoggerFactory:
    @staticmethod
    def get_final_decorated_logger(cfg: DictConfig):
        log = cfg.logging

        return WandbLoggerDecorator(
            MLFlowLogger(log.get("mlflow", None)), log.get("wandb", None)
        )
