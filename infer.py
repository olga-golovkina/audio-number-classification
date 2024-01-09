import glob
from os import path
from pathlib import Path

import pandas as pd
import torchaudio
from hydra import compose, initialize
from tqdm import tqdm

from audio_number_classification.factoring.augmentation import AugmentationFactory
from audio_number_classification.factoring.model import ModelFactory
from audio_number_classification.predictor import AudioPredictor


def main():
    initialize(
        version_base=None, config_path="configs", job_name="audio_number_classification"
    )

    cfg = compose(config_name="config")
    test_dataset_folder = cfg["path"]["test"]

    trans = AugmentationFactory.get_augmentation(cfg)
    model = ModelFactory.get_model_from(cfg)
    _ = model

    test_audio_path = glob.glob(path.join(test_dataset_folder, "*.mp3"))

    audio_results = []

    for i in tqdm(range(len(test_audio_path))):
        file = test_audio_path[i]
        audio, sample_rate = torchaudio.load(file)
        audio = trans.transform(audio, sample_rate)

        audio_results.append(
            {"row ID": i, "answer": AudioPredictor.predict(model, audio)}
        )

    results_df = pd.DataFrame(audio_results)

    res_path = Path(cfg["path"]["result"])
    res_folder = res_path.parents[0]

    if not res_folder.exists():
        res_folder.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(cfg["path"]["result"], index=False, sep=",")


if __name__ == "__main__":
    main()
