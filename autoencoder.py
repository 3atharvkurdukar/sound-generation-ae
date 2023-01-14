import tensorflow as tf
import numpy as np
import os
import pickle

from keras import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, \
    Dense, Reshape, Conv2DTranspose, Activation, Lambda
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras import backend as K

tf.compat.v1.disable_eager_execution()


class Autoencoder:
    """
    Autoencoder represents a deep convolutional autoencoder architectire with
    mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape: tuple,
                 conv_filters: tuple,
                 conv_kernels: tuple,
                 conv_strides: tuple,
                 latent_space_dims: tuple) -> None:
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dims = latent_space_dims

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate: float):
        optimizer = Adam(lr=learning_rate)
        loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)

    def save(self, save_folder="."):
        self._create_folder_if_necessary(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    def generate(self, latent_representations):
        return self.decoder.predict(latent_representations)

    @classmethod
    def load(cls, save_folder="."):
        params_path = os.path.join(save_folder, 'params.pkl')
        with open(params_path, "rb") as f:
            parameters = pickle.load(f)
            autoencoder = cls(*parameters)
        weights_path = os.path.join(save_folder, 'weights.h5')
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _create_folder_if_necessary(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    def _save_parameters(self, save_folder):
        filepath = os.path.join(save_folder, 'params.pkl')
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dims
        ]
        with open(filepath, 'wb') as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        filepath = os.path.join(save_folder, 'weights.h5')
        self.model.save_weights(filepath)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name='encoder')

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name='encoder_input')

    def _add_conv_layers(self, encoder_input):
        """Create the convolutional blocks in the encoder."""
        x = encoder_input
        for i in range(self._num_conv_layers):
            x = self._add_conv_layer(x, i)
        return x

    def _add_conv_layer(self, x, layer_index: int):
        """
        Create a single convolutional block in the encoder consisting of
        Conv2D + ReLU + Batch Normalisation.
        """
        layer_num = layer_index + 1
        x = Conv2D(filters=self.conv_filters[layer_index],
                   kernel_size=self.conv_kernels[layer_index],
                   strides=self.conv_strides[layer_index],
                   padding='same',
                   name=f"encoder_conv_{layer_num}")(x)
        x = ReLU(name=f"encoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_num}")(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add the bottleneck layer (dense layer)."""
        self._shape_before_bottleneck = K.int_shape(
            x)[1:]  # Ignore the batch size dimension
        x = Flatten(name='encoder_flatten')(x)
        x = Dense(units=self.latent_space_dims, name='encoder_output')(x)
        return x

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name='decoder')

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dims, name='decoder_input')

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name='decoder_dense')(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Create the convolutional transpose blocks in the decoder."""
        # Loop through all the conv layers in the reverse order and stop at the
        # first layer
        for i in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(x, i)
        return x

    def _add_conv_transpose_layer(self, x, layer_index: int):
        """
        Create a single convolutional transpose block in the decoder consisting of
        Conv2DTranspose + ReLU + Batch Normalisation.
        """
        layer_num = self._num_conv_layers - layer_index
        x = Conv2DTranspose(filters=self.conv_filters[layer_index],
                            kernel_size=self.conv_kernels[layer_index],
                            strides=self.conv_strides[layer_index],
                            padding='same',
                            name=f"decoder_conv_transpose_{layer_num}")(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        x = Conv2DTranspose(filters=1,
                            kernel_size=self.conv_kernels[0],
                            strides=self.conv_strides[0],
                            padding='same',
                            name=f"decoder_conv_transpose_{self._num_conv_layers}")(x)
        output_layer = Activation('sigmoid', name='sigmoid_layer')(x)
        return output_layer

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name='autoencoder')


def calculate_reconstruction_loss(y_target, y_predicted):
    error = y_target - y_predicted
    return K.mean(K.square(error), axis=[1, 2, 3])


def calculate_kl_loss(autoencoder):
    # wrap `_calculate_kl_loss` such that it takes the model as an argument,
    # returns a function which can take arbitrary number of arguments
    # (for compatibility with `metrics` and utility in the loss function)
    # and returns the kl loss
    def _calculate_kl_loss(*args):
        kl_loss = -0.5 * K.sum(1 + autoencoder.log_variance - K.square(autoencoder.mu) -
                               K.exp(autoencoder.log_variance), axis=1)
        return kl_loss
    return _calculate_kl_loss


class VAE(Autoencoder):
    """
    VAE represents a deep convolutional variational autoencoder architectire with
    mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape: tuple,
                 conv_filters: tuple,
                 conv_kernels: tuple,
                 conv_strides: tuple,
                 latent_space_dims: tuple) -> None:
        self.mu = None
        self.log_variance = None
        self.reconstruction_loss_weight = 1000
        super().__init__(input_shape,
                         conv_filters,
                         conv_kernels,
                         conv_strides,
                         latent_space_dims)

    def compile(self, learning_rate: float):
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss,
                           metrics=[calculate_reconstruction_loss,
                                    calculate_kl_loss(self)])

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = calculate_reconstruction_loss(
            y_target, y_predicted)
        kl_loss = calculate_kl_loss(self)()
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def _add_bottleneck(self, x):
        """Flatten data and add the bottleneck layer with Gaussian sampling (dense layer)."""
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        # Ignore the batch size dimension
        x = Flatten(name='encoder_flatten')(x)

        self.mu = Dense(units=self.latent_space_dims, name='mu')(x)
        self.log_variance = Dense(
            units=self.latent_space_dims, name='log_variance')(x)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        x = Lambda(sample_point_from_normal_distribution,
                   name='encoder_output')([self.mu, self.log_variance])
        return x


if __name__ == '__main__':
    autoencoder = VAE(input_shape=(28, 28, 1),
                      conv_filters=(32, 64, 64, 64),
                      conv_kernels=(3, 3, 3, 3),
                      conv_strides=(1, 2, 2, 1),
                      latent_space_dims=2)
    autoencoder.summary()
