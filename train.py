import numpy as np
import os

from keras.datasets import mnist
from autoencoder import VAE

SPECTROGRAMS_PATH = 'datasets/fsdd/spectrograms'
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 150


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1, ))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1, ))
    return x_train, y_train, x_test, y_test


def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, files in os.walk(spectrograms_path):
        for file in files:
            if file.endswith('.npy'):
                spectrogram = np.load(os.path.join(
                    root, file))  # (n_bins, n_frames)
                x_train.append(spectrogram)
                file_paths.append(os.path.join(root, file))
    x_train = np.array(x_train)  # (n_samples, n_bins, n_frames)
    x_train = x_train[..., np.newaxis]  # (n_samples, n_bins, n_frames, 1)
    return x_train, file_paths


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(input_shape=(256, 64, 1),
                      conv_filters=(512, 256, 128, 64, 32),
                      conv_kernels=(3, 3, 3, 3, 3),
                      conv_strides=(2, 2, 2, 2, (2, 1)),
                      latent_space_dims=128)
    autoencoder.reconstruction_loss_weight = 1000000
    autoencoder.summary()
    autoencoder.compile(learning_rate=learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == '__main__':
    x_train, _ = load_fsdd(SPECTROGRAMS_PATH)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save('fsdd_vae')
    autoencoder2 = VAE.load('fsdd_vae')
    autoencoder2.summary()
