import numpy as np
import matplotlib.pyplot as plt

from autoencoder import VAE
from train import load_mnist


def select_images(images, labels, num_images=10):
    select_images_indices = np.random.choice(range(len(images)), num_images)
    sample_images = images[select_images_indices]
    sample_labels = labels[select_images_indices]
    return sample_images, sample_labels


def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (img, rec_img) in enumerate(zip(images, reconstructed_images)):
        img = img.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(img, cmap="gray_r")
        rec_img = rec_img.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(rec_img, cmap="gray_r")
    plt.show()


def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                c=sample_labels,
                cmap='rainbow',
                alpha=0.5,
                s=4)
    plt.colorbar()
    plt.show()


def plot_generated_images(generated_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(generated_images)
    for i, img in enumerate(generated_images):
        img = img.squeeze()
        ax = fig.add_subplot(1, num_images, i + 1)
        ax.axis("off")
        ax.imshow(img, cmap="gray_r")
    plt.show()


if __name__ == '__main__':
    autoencoder = VAE.load('mnist_vae')
    x_train, y_train, x_test, y_test = load_mnist()

    num_sample_images = 12
    sample_images, sample_labels = select_images(x_test,
                                                 y_test,
                                                 num_sample_images)
    reconstructed_images, latent_representations = autoencoder.reconstruct(
        sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images)

    num_sample_images = 15000
    sample_images, sample_labels = select_images(x_test,
                                                 y_test,
                                                 num_sample_images)
    reconstructed_images, latent_representations = autoencoder.reconstruct(
        sample_images)
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)
