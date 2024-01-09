from abc import ABC, abstractmethod

import git


class Logger(ABC):
    @abstractmethod
    def log_hyperparams(self, hyperparams, version_code):
        pass

    @abstractmethod
    def log_metrics(self, metric):
        pass

    @abstractmethod
    def close(self):
        pass

    def get_code_version(self):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.commit.hexsha

        return repo.git.rev_parse(sha, short=7)


class LoggerDecorator(Logger, ABC):
    def log_metrics(self, metric):
        self.logger.log_metrics(metric)

    def log_hyperparams(self, hyperparams, version_code):
        self.logger.log_hyperparams(hyperparams, version_code)

    def __init__(self, logger: Logger):
        self.logger = logger
