import librosa

from preprocessing import MinMaxNormalizer


class SoundGenerator:
    """
    SoundGenerator is responsible for generating audios from spectrograms
    """

    def __init__(self, model, hop_length):
        self.model = model
        self.hop_length = hop_length
        self._min_max_normalizer = MinMaxNormalizer(0, 1)

    def generate(self, spectrograms, min_max_values):
        """
        Generate audios from spectrograms
        :param spectrograms: spectrograms to generate audios from
        :param min_max_values: min max values of the spectrograms
        :return: generated audios
        """
        gen_spectrograms, latent_representations = self.model.reconstruct(
            spectrograms)
        audios = self.convert_spectrograms_to_audios(
            gen_spectrograms, min_max_values)
        return audios, latent_representations

    def convert_spectrograms_to_audios(self, spectrograms, min_max_values):
        """
        Convert spectrograms to audios
        :param spectrograms: spectrograms to convert
        :param min_max_values: min max values of the spectrograms
        :return: converted audios
        """
        audios = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            log_spectrogram = spectrogram[:, :, 0]
            denorm_log_spectrograms = self._min_max_normalizer.denormalise(
                log_spectrogram, min_max_value['min'], min_max_value['max'])
            audio = librosa.istft(denorm_log_spectrograms, self.hop_length)
            audios.append(audio)
        return audios
