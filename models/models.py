import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, MaxPooling2D, Dropout,
    Input, concatenate, Cropping2D
)

from keras_unet.models.custom_unet import conv2d_block
from keras_unet.models.vanilla_unet import get_crop_shape


def standard_unet(input_shape, num_classes=1):

    inputs = Input(shape=input_shape)
    x = inputs

    descending_layers = []
    num_filters = 64

    # Encoding
    for i in range(4):
        x = conv2d_block(inputs=x, filters=num_filters,
                         use_batch_norm=False, dropout=0.0, padding='valid')
        descending_layers.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        num_filters *= 2

    dropout = 0.5
    x = Dropout(dropout)(x)
    x = conv2d_block(inputs=x, filters=num_filters,
                     use_batch_norm=False, dropout=0.0, padding='valid')

    # Decoding
    for layer in reversed(descending_layers):
        num_filters //= 2
        x = Conv2DTranspose(filters=num_filters, kernel_size=(2, 2),
                            strides=(2, 2), padding='valid')(x)

        ch, cw = get_crop_shape(int_shape(layer), int_shape(x))
        layer = Cropping2D(cropping=(ch, cw))(layer)

        x = concatenate([x, layer])
        x = conv2d_block(inputs=x, filters=num_filters,
                         use_batch_norm=False, dropout=0.0, padding='valid')

    output_activation = 'sigmoid' if num_classes == 1 else 'softmax'
    outputs = Conv2D(filters=num_classes, kernel_size=(1, 1),
                     activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def vgg19_unet(input_shape, num_classes=1):
    pass