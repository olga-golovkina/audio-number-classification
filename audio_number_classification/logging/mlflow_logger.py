from omegaconf import DictConfig

from audio_number_classification.logging.logger import Logger


class MLFlowLogger(Logger):
    def close(self):
        pass

    def __init__(self, cfg: DictConfig):
        pass

    def log_metrics(self, metric):
        super().log_metrics(metric)

    def log_hyperparams(self, hyperparams):
        super().log_hyperparams(hyperparams)
