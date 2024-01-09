from omegaconf import DictConfig

from audio_number_classification.logging.logger import Logger


class MLFlowLogger(Logger):
    def __init__(self, cfg: DictConfig):
        pass

    def log_exception(self, exception):
        super().log_exception(exception)

    def log_metrics(self, metric):
        super().log_metrics(metric)

    def log_hyperparams(self, hyperparams, version_code):
        super().log_hyperparams(hyperparams, version_code)
