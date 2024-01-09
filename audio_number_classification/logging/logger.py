from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def log_hyperparams(self, hyperparams, version_code):
        pass

    @abstractmethod
    def log_metrics(self, metric):
        pass

    @abstractmethod
    def log_exception(self, exception):
        pass


class LoggerDecorator(Logger, ABC):
    def log_metrics(self, metric):
        self.logger.log_metrics(metric)

    def log_hyperparams(self, hyperparams, version_code):
        self.logger.log_hyperparams(hyperparams, version_code)

    def log_exception(self, exception):
        self.logger.log_exception(exception)

    def __init__(self, logger: Logger):
        self.logger = logger


class WandbLoggerDecorator(LoggerDecorator):
    def __init__(self, logger, cfg):
        super().__init__(logger)

    def log_exception(self, exception):
        super().log_exception(exception)

    def log_metrics(self, metric):
        super().log_metrics(metric)

    def log_hyperparams(self, hyperparams, version_code):
        super().log_hyperparams(hyperparams, version_code)


class MLFlowLoggerDecorator(LoggerDecorator):
    def __init__(self, logger, cfg):
        super().__init__(logger)

    def log_exception(self, exception):
        super().log_exception(exception)

    def log_metrics(self, metric):
        super().log_metrics(metric)

    def log_hyperparams(self, hyperparams, version_code):
        super().log_hyperparams(hyperparams, version_code)


class FileLogger(Logger):
    def log_hyperparams(self, hyperparams, version_code):
        pass

    def log_metrics(self, metric):
        pass

    def log_exception(self, exception):
        pass
