import torchaudio
from omegaconf import DictConfig

from audio_number_classification.augmentation import AudioTransformer


class AugmentationFactory:
    @staticmethod
    def get_augmentation(cfg: DictConfig):
        device = cfg["shared"]["device"]

        sample_rate = cfg["dataset"]["sample_rate"]
        sample_size = cfg["dataset"]["sample_size"]

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, **cfg["spectogram"]
        )

        return AudioTransformer(mel_spectrogram, sample_rate, sample_size, device)
