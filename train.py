from pathlib import Path

from hydra import compose, initialize

from audio_number_classification.factoring.augmentation import AugmentationFactory
from audio_number_classification.factoring.dataset import DatasetFactory
from audio_number_classification.factoring.logger import LoggerFactory
from audio_number_classification.factoring.model import ModelFactory
from audio_number_classification.factoring.trainer import TrainerFactory


def main():
    initialize(
        version_base=None, config_path="configs", job_name="audio_number_classification"
    )
    cfg = compose(config_name="config")

    trans = AugmentationFactory.get_augmentation(cfg)

    train_dataset = DatasetFactory.get_dataset(trans, cfg, Path(cfg["path"]["train"]))
    val_dataset = DatasetFactory.get_dataset(trans, cfg, Path(cfg["path"]["val"]))

    model = ModelFactory.create_empty_model(cfg)

    trainer = TrainerFactory.get_custom_trainer(
        model,
        train_dataset,
        val_dataset,
        LoggerFactory.get_final_decorated_logger(cfg),
        cfg,
    )

    best_model, _, _ = trainer.fit()

    best_model.save(Path(cfg["path"]["model"]))


if __name__ == "__main__":
    main()
