from keras import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense
from keras import backend as K


class AutoEncoder:
    """
    AutoEncoder represents a deep convolutional autoencoder architectire with
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

        self._build()

    def summary(self):
        self.encoder.summary()
        # self.decoder.summary()
        # self.model.summary()

    def _build(self):
        self._build_encoder()
        # self._build_decoder()
        # self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.encoder = Model(encoder_input, bottleneck, name='encoder')
        pass

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name='encoder_input')

    def _add_conv_layers(self, encoder_input):
        """Create the convolutional blocks in the encoder."""
        x = encoder_input
        for i in range(self._num_conv_layers):
            x = self._add_conv_layer(x, i)
        return x

    def _add_conv_layer(self, x, layer_index):
        """
        Create a single convolutional block in the encoder consisting of
        Conv2D + ReLU + Batch Normalisation.
        """
        x = Conv2D(filters=self.conv_filters[layer_index],
                   kernel_size=self.conv_kernels[layer_index],
                   strides=self.conv_strides[layer_index],
                   padding='same',
                   name=f"encoder_conv_{layer_index + 1}")(x)
        x = ReLU(name=f"encoder_relu_{layer_index + 1}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_index + 1}")(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add the bottleneck layer (dense layer)."""
        self._shape_before_bottleneck = K.int_shape(
            x)[1:]  # Ignore the batch size dimension
        x = Flatten(name='encoder_flatten')(x)
        x = Dense(units=self.latent_space_dims, name='encoder_output')(x)
        return x

    def _build_decoder(self):
        pass

    def _build_autoencoder(self):
        pass


if __name__ == '__main__':
    autoencoder = AutoEncoder(input_shape=(28, 28, 1),
                              conv_filters=(32, 64, 64, 64),
                              conv_kernels=(3, 3, 3, 3),
                              conv_strides=(1, 2, 2, 1),
                              latent_space_dims=2)
    autoencoder.summary()
