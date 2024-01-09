import mlflow
from omegaconf import DictConfig

from .logger import Logger


class MLFlowLogger(Logger):
    def close(self):
        mlflow.end_run()

    def __init__(self, cfg: DictConfig):
        mlflow.set_tracking_uri(cfg.get("uri", "128.0.1.1:8080"))
        mlflow.log_params({"version_code": self.get_code_version()})

    def log_metrics(self, metric):
        mlflow.log_metrics(metric, step=metric["epoch"])

    def log_hyperparams(self, hyperparams: dict):
        mlflow.log_params(hyperparams)
