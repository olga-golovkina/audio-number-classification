from pathlib import Path

import torchaudio

from audio_number_classification.augmentation import AudioTransformer
from audio_number_classification.dataset import AudioDataset


def main():
    train_annotations = "data/train_annotations.csv"

    SAMPLE_RATE = 22050
    SAMPLE_SIZE = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64, center=True
    )

    _ = AudioDataset(
        Path(train_annotations),
        AudioTransformer(mel_spectrogram, SAMPLE_RATE, SAMPLE_SIZE, "cpu"),
        "cpu",
        True,
    )


if __name__ == "__main__":
    main()
