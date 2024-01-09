from audio_number_classification.logging.logger import LoggerDecorator


class WandbLoggerDecorator(LoggerDecorator):
    def __init__(self, logger, cfg):
        super().__init__(logger)

    def log_exception(self, exception):
        super().log_exception(exception)

    def log_metrics(self, metric):
        super().log_metrics(metric)

    def log_hyperparams(self, hyperparams, version_code):
        super().log_hyperparams(hyperparams, version_code)
