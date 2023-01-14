import librosa
import os
import numpy as np
import pickle


class Loader:
    """
    Loader is responsible for loading an audio file
    """

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal, _ = librosa.load(file_path,
                                 sr=self.sample_rate,
                                 duration=self.duration,
                                 mono=self.mono)
        return signal


class Padder:
    """
    Padder is responsible for padding an array
    """

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (num_missing_items, 0), self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (0, num_missing_items), self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """
    LogSpectrogramExtractor is responsible for extracting log spectrograms
    (in dB) from a time series signal
    """

    def __init__(self, frame_size, hop_length) -> None:
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]  # Output shape is like: (1  + frame_size/2, num_frames)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class MinMaxNormaliser:
    """
    MinMaxNormaliser is responsible for normalising an array
    """

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array):
        normalised_array = (array - self.min) / (self.max - self.min)
        normalised_array = normalised_array * (self.max - self.min) + self.min
        return normalised_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


class Saver:
    def __init__(self, feature_save_dir, min_max_values_save_dir) -> None:
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_file_path(file_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, feature)

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(
            self.min_max_values_save_dir, 'min_max_values.pkl')
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def _generate_file_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + '.npy')
        return save_path


class PreprocessingPipeline:
    """
    PreprocessingPipleline processes the audio files in a directory, applying
    the following steps to each file:
    1. Load the audio file
    2. Pad the signal (if necessary)
    3. Extract the log spectrogram
    4. Normalise the spectrogram
    5. Save the normalised spectrogram
    6. Store the min max values for all the spectrograms
    """

    def __init__(self) -> None:
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}

        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir):
        print(audio_files_dir)
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f'Processed file: {file_path}')
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_values(save_path, feature.min(), feature.max())
        return norm_feature

    def _is_padding_necessary(self, signal):
        self._num_expected_samples = int(
            self.loader.sample_rate * self.loader.duration)
        return len(signal) < self._num_expected_samples

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        signal = self.padder.right_pad(signal, num_missing_samples)
        return signal

    def _store_min_max_values(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }


if __name__ == '__main__':
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAM_SAVE_DIR = 'dataset/fsdd/spectrograms'
    MIN_MAX_VALUES_SAVE_DIR = 'datasets/fsdd'
    FILES_DIR = 'daasets/fsdd/audio'

    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTROGRAM_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    pipeline = PreprocessingPipeline()
    pipeline.loader = loader
    pipeline.padder = padder
    pipeline.extractor = log_spectrogram_extractor
    pipeline.normaliser = min_max_normaliser
    pipeline.saver = saver

    pipeline.process(FILES_DIR)
