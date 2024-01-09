import torch
import torchaudio.transforms as trans


class AudioTransformer:
    def __init__(self, transformation, sample_rate, sample_size, device):
        self.transformation = transformation.to(device)
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.device = device

    def __mix_signal(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=0)

        return signal

    def __resample(self, signal, sample_rate):
        if sample_rate != self.sample_rate:
            resampler = trans.Resample(sample_rate, self.sample_rate)
            signal = resampler(signal)

        return signal

    def __right_pad(self, signal):
        signal_len = signal.shape[1]

        if signal_len < self.sample_size:
            missing_size = self.sample_size - signal_len
            padded = (0, missing_size)
            signal = torch.nn.functional.pad(signal, padded)

        return signal

    def __cut(self, signal):
        if signal.shape[1] > self.sample_size:
            signal = signal[:, : self.sample_size]

        return signal

    def transform(self, signal, sample_rate):
        signal = signal.to(self.device)
        signal = self.__resample(signal, sample_rate)
        signal = self.__mix_signal(signal)
        signal = self.__cut(signal)
        signal = self.__right_pad(signal)

        signal = self.transformation(signal)

        return signal
