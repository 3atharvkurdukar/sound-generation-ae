import os
import numpy as np
import pickle
import soundfile as sf
from tqdm import tqdm

from train import load_fsdd
from soundgenerator import SoundGenerator
from autoencoder import VAE
from preprocessing import Loader

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = 'samples/original'
SAVE_DIR_GENERATED = 'samples/generated'
MIN_MAX_VALUES_PATH = 'datasets/fsdd/min_max_values.pkl'
SPECTROGRAMS_PATH = 'datasets/fsdd/spectrograms'


def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_samples=2):
    sampled_indices = np.random.choice(len(spectrograms), num_samples)
    sampled_spectrograms = spectrograms[sampled_indices]
    file_paths = [file_paths[i] for i in sampled_indices]
    sampled_min_max_values = [min_max_values[file_path]
                              for file_path in file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrograms, sampled_min_max_values


def save_samples(signals, save_dir, sampling_rate=22050):
    os.makedirs(save_dir, exist_ok=True)
    print(f'Saving samples to {save_dir}...')
    for i, sample in tqdm(enumerate(signals)):
        save_path = os.path.join(save_dir, f'{i}.wav')
        sf.write(save_path, sample, sampling_rate)


if __name__ == '__main__':
    # Initialize the sound generator
    autoencoder = VAE.load('fsdd_vae')
    sound_generator = SoundGenerator(autoencoder, HOP_LENGTH)

    # Load spectrograms and min-max values
    with open(MIN_MAX_VALUES_PATH, 'rb') as f:
        min_max_values = pickle.load(f)
    spectrograms, file_paths = load_fsdd(SPECTROGRAMS_PATH)
    print(f'Loaded {len(spectrograms)} spectrograms.')

    # Sample spectrograms and min-max values
    sampled_spectrograms, sampled_min_max_values = select_spectrograms(
        spectrograms, file_paths, min_max_values, 5)
    print(f'Sampled {len(sampled_spectrograms)} spectrograms.')

    # Generate audio signals
    signals, _ = sound_generator.generate(sampled_spectrograms,
                                          sampled_min_max_values)
    print('Generated audio signals.')

    # Convert spectrograms to audio signals
    original_samples = sound_generator.convert_spectrograms_to_audios(sampled_spectrograms,
                                                                      sampled_min_max_values)
    print('Converted original spectrograms to audio signals.')

    # Save audio signals
    save_samples(signals, SAVE_DIR_GENERATED)
    save_samples(original_samples, SAVE_DIR_ORIGINAL)
