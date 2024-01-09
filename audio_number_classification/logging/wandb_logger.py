import wandb
from audio_number_classification.logging.logger import LoggerDecorator


class WandbLoggerDecorator(LoggerDecorator):
    def close(self):
        wandb.finish()

        self.logger.close()

    def __init__(self, logger, cfg):
        super().__init__(logger)

        wandb.init(
            project=cfg.project, anonymous=cfg.get("anonymous", "must"), notes=cfg.notes
        )

    def log_metrics(self, metric: dict):
        super().log_metrics(metric)
        wandb.log(metric)

    def log_hyperparams(self, hyperparams: dict, version_code):
        super().log_hyperparams(hyperparams, version_code)

        wandb.log(version_code)
        wandb.log(hyperparams)
