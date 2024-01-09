import wandb
from audio_number_classification.logging.logger import LoggerDecorator


class WandbLoggerDecorator(LoggerDecorator):
    def close(self):
        wandb.finish()

        self.logger.close()

    def __init__(self, logger, cfg):
        super().__init__(logger)

        wandb.init(
            project=cfg.project,
            anonymous=cfg.get("anonymous", "must"),
            config={"version_code": self.get_code_version()},
            notes=cfg.notes,
        )

    def log_metrics(self, metric: dict):
        super().log_metrics(metric)
        wandb.log(metric)

    def log_hyperparams(self, hyperparams: dict):
        super().log_hyperparams(hyperparams)

        wandb.log(hyperparams)
