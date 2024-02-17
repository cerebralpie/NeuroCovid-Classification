import tensorflow as tf

import src.utils as nc_utils
from tensorflow.keras import Input
# from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv2D, UpSampling2D, MaxPooling2D, Concatenate, BatchNormalization,
    Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import (
    DenseNet201
)


def _conv2d_block(
        inputs: tf.Tensor,
        num_filters: int,
        kernel_size: tuple[int, int] = (3, 3),
        padding: str = 'same'
) -> tf.Tensor:
    """
    Create a convolutional block consisting of two convolutional layers with
    optional batch normalization and dropout.

    Args:
        inputs: The input tensor to the block.
        num_filters: The number of filters for each convolutional layer.
        kernel_size: The kernel size for each convolutional layer. Defaults to
                     (3, 3).
        padding: The padding to use for the convolutional layers. Defaults to
                 'same'.

    Returns:
        The output tensor of the convolutional block.
    """
    # First convolutional layer
    x = Conv2D(filters=num_filters,
               kernel_size=kernel_size,
               padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Second convolutional layer
    x = Conv2D(filters=num_filters,
               kernel_size=kernel_size,
               padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def unet_model(
        input_shape: tuple[int, int, int],
        num_classes: int = 1,
        num_filters: int = 16,
        num_layers: int = 4,
        augment_data: bool = False
) -> Model:
    """
    Create a U-Net model for image segmentation.

    This function implements a U-Net model suitable for image segmentation
    tasks. It uses a standard encoder-decoder architecture with skip connections
    between corresponding layers to preserve spatial information. Data
    augmentation can be applied before feeding the input into the network.

    Args:
        input_shape: A tuple representing the shape of the input images.
        num_classes: The number of segmentation classes (including background).
                     Defaults to 1.
        num_filters: The initial number of filters in the first convolutional
                     layer. Defaults to 16.
        num_layers: The number of encoding/decoding layers (excluding
                    bottleneck). Defaults to 4.
        augment_data: Whether to apply data augmentation to the input dataset.
                      Defaults to False.

    Returns:
        A TensorFlow keras Model instance representing the U-Net model.
    """
    if num_classes > 1:
        output_activation = "softmax"
    elif num_classes == 1:
        output_activation = "sigmoid"
    else:
        raise ValueError("Invalid number of classes")

    inputs = Input(shape=input_shape)
    x = inputs

    if augment_data:
        x = nc_utils.get_data_augmentation_pipeline()(x)

    # Encoding layers
    skip_connections = []
    for i in range(num_layers):
        x = _conv2d_block(inputs=x, num_filters=num_filters)
        skip_connections.append(x)

        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        num_filters = num_filters * 2

    # Bottleneck
    x = _conv2d_block(inputs=x, num_filters=num_filters)

    # Decoding layers
    for skip_connection in reversed(skip_connections):
        num_filters = num_filters // 2

        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate()([x, skip_connection])

        x = _conv2d_block(inputs=x, num_filters=num_filters)

    x = Conv2D(filters=num_classes, kernel_size=(1, 1), padding="same")(x)
    x = Activation(output_activation)(x)

    model = Model(inputs=inputs, outputs=x)

    return model


def densenet_unet_model(
        input_shape: tuple[int, int, int],
        num_classes: int = 1,
        augment_data: bool = False
) -> Model:
    """
    Create a U-Net model for image segmentation.

    This function implements a U-Net model suitable for image segmentation
    tasks. It uses a standard encoder-decoder architecture with skip connections
    between corresponding layers to preserve spatial information. Data
    augmentation can be applied before feeding the input into the network.

    Args:
        input_shape: A tuple representing the shape of the input images.
        num_classes: The number of segmentation classes (including background).
                     Defaults to 1.
        num_filters: The initial number of filters in the first convolutional
                     layer. Defaults to 16.
        num_layers: The number of encoding/decoding layers (excluding
                    bottleneck). Defaults to 4.
        augment_data: Whether to apply data augmentation to the input dataset.
                      Defaults to False.

    Returns:
        A TensorFlow keras Model instance representing the U-Net model.
    """
    if num_classes > 1:
        output_activation = "softmax"
    elif num_classes == 1:
        output_activation = "sigmoid"
    else:
        raise ValueError("Invalid number of classes")

    inputs = Input(shape=input_shape)
    x = inputs

    if augment_data:
        x = nc_utils.get_data_augmentation_pipeline()(x)

    # Encoding layers
    encoder = DenseNet201(input_tensor=x, weights="imagenet", include_top=False)
    skip_connection_names = ["input_1", "conv1/relu", "pool2_relu", "pool3_relu", "pool4_relu"]
    encoder_output = encoder.get_layer("relu").output
    num_layers = len(skip_connection_names)

    skip_connections = []
    for i in range(num_layers):
        x_skip = encoder.get_layer(skip_connection_names[i]).output
        skip_connections.append(x_skip)

    # Bottleneck
    num_filters = 16 + (2 ** (num_layers - 1))
    x = encoder_output
    x = _conv2d_block(inputs=x, num_filters=num_filters)

    # Decoding layers
    for skip_connection in reversed(skip_connections):
        num_filters = num_filters // 2

        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate()([x, skip_connection])

        x = _conv2d_block(inputs=x, num_filters=num_filters)

    outputs = Conv2D(filters=num_classes,
                     kernel_size=(1, 1),
                     padding="same",
                     activation=output_activation)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


# def _conv2d_block(
#         inputs: tf.Tensor,
#         use_batch_normalization: bool = True,
#         dropout_rate: float = 0.3,
#         dropout_type: str = 'spatial',
#         num_filters: int = 16,
#         kernel_size: tuple[int, int] = (3, 3),
#         activation: str = 'relu',
#         kernel_initializer: str = 'he_normal',
#         padding: str = 'same'
# ) -> tf.Tensor:
#     """
#     Create a convolutional block consisting of two convolutional layers with
#     optional batch normalization and dropout.
#
#     Args:
#         inputs: The input tensor to the block.
#         use_batch_normalization: Whether to apply batch normalization after each
#                                  convolution. Defaults to True.
#         dropout_rate: The dropout rate to apply. Set to 0.0 to disable dropout.
#                       Defaults to 0.3.
#         dropout_type: The type of dropout to apply, either 'spatial' or
#                       'standard'. Defaults to 'spatial'.
#         num_filters: The number of filters for each convolutional layer.
#                      Defaults to 16.
#         kernel_size: The kernel size for each convolutional layer. Defaults to
#                      (3, 3).
#         activation: The activation function to use after each convolution.
#                     Defaults to 'relu'.
#         kernel_initializer: The kernel initializer for the convolutional layers.
#                             Defaults to 'he_normal'.
#         padding: The padding to use for the convolutional layers. Defaults to
#                  'same'.
#
#     Returns:
#         The output tensor of the convolutional block
#
#     Raises:
#         ValueError: If the dropout_type is not 'spatial' or 'standard'.
#     """
#     if dropout_type == 'spatial':
#         drop = SpatialDropout2D
#     elif dropout_type == 'standard':
#         drop = Dropout
#     else:
#         raise ValueError(
#             f"dropout_type must be one of ['spatial', 'standard'] got "
#             f"{dropout_type} instead."
#         )
#
#     # First convolutional layer
#     x = Conv2D(filters=num_filters,
#                kernel_size=kernel_size,
#                activation=activation,
#                kernel_initializer=kernel_initializer,
#                padding=padding,
#                use_bias=not use_batch_normalization)(inputs)
#
#     if use_batch_normalization:
#         x = BatchNormalization()(x)
#
#     if dropout_rate > 0.0:
#         x = drop(dropout_rate)(x)
#
#     # Second convolutional layer
#     x = Conv2D(filters=num_filters,
#                kernel_size=kernel_size,
#                activation=activation,
#                kernel_initializer=kernel_initializer,
#                padding=padding,
#                use_bias=not use_batch_normalization)(x)
#
#     if use_batch_normalization:
#         x = BatchNormalization()(x)
#
#     return x


# def _get_cropping_dimensions(
#         target_tensor: tf.Tensor,
#         reference_tensor: tf.Tensor
# ) -> tuple[tuple[int, int], tuple[int, int]] | tuple[None, None]:
#     """
#     Calculate the cropping dimensions needed to align a target tensor with a
#     reference tensor.
#
#     This function calculates the top and bottom (height) and left and right
#     (width) cropping dimensions required to make the 'target' tensor have the
#     same spatial dimensions (height, width) as the 'reference' tensor. It
#     assumes that the depth (channels) and batch dimensions are already
#     compatible.
#
#     Args:
#         target_tensor: The tensor to be cropped.
#         reference_tensor: The reference tensor to align with.
#
#     Returns:
#         A tuple of two tuples, representing the cropping dimensions for height
#         and width, respectively. Each inner tuple contains the starting and
#         ending indices for cropping.
#
#     Raises:
#         ValueError: If the target tensor has smaller height or width than the
#                     reference tensor.
#     """
#     target_shape = K.int_shape(target_tensor)
#     reference_shape = K.int_shape(reference_tensor)
#
#     target_height, target_width = target_shape[1], target_shape[2]
#     reference_height, reference_width = reference_shape[1], reference_shape[2]
#
#     total_height_crop = target_height - reference_height
#     total_width_crop = target_width - reference_width
#
#     if total_height_crop == 0 and total_width_crop == 0:
#         return None, None
#
#     if total_height_crop <= 0 and total_width_crop <= 0:
#         raise ValueError("The target tensor has smaller height and width than "
#                          "the reference tensor.")
#     elif total_height_crop <= 0:
#         raise ValueError("The target tensor has smaller height than the "
#                          "reference tensor.")
#     elif total_width_crop <= 0:
#         raise ValueError("The target tensor has smaller width than the "
#                          "reference tensor.")
#
#     if total_height_crop % 2 == 0:
#         top_crop, bottom_crop = total_height_crop // 2, total_height_crop // 2
#     else:
#         top_crop, bottom_crop = (total_height_crop // 2, (total_height_crop //
#                                  2) + 1)
#
#     if total_width_crop % 2 == 0:
#         left_crop, right_crop = total_width_crop // 2, total_width_crop // 2
#     else:
#         left_crop, right_crop = total_width_crop // 2, total_width_crop // 2 + 1
#
#     return (top_crop, bottom_crop), (left_crop, right_crop)


# def unet_model(
#         input_shape: tuple[int, int, int],
#         num_classes: int = 1,
#         dropout_rate: float = 0.5,
#         num_filters: int = 16,
#         num_layers: int = 4,
#         output_activation: str = 'sigmoid',
#         augment_data: bool = False
# ) -> tf.keras.models.Model:
#     """
#     Create a U-Net model for image segmentation.
#
#     This function implements a U-Net model suitable for image segmentation
#     tasks. It uses a standard encoder-decoder architecture with skip connections
#     between corresponding layers to preserve spatial information. Data
#     augmentation can be applied before feeding the input into the network.
#
#     Args:
#         input_shape: A tuple representing the shape of the input images.
#         num_classes: The number of segmentation classes (including background).
#                      Defaults to 1.
#         dropout_rate: The dropout rate to apply for regularization. Defaults to
#                       0.5.
#         num_filters: The initial number of filters in the first convolutional
#                      layer. Defaults to 16.
#         num_layers: The number of encoding/decoding layers (excluding
#                     bottleneck). Defaults to 4.
#         output_activation: The activation function to use for the final output
#                            layer. Defaults to 'sigmoid'.
#         augment_data: Whether to apply data augmentation to the input dataset.
#                       Defaults to False.
#
#     Returns:
#         A TensorFlow keras Model instance representing the U-Net model.
#     """
#     inputs = Input(shape=input_shape)
#     x = inputs
#
#     if augment_data:
#         x = nc_utils.get_data_augmentation_pipeline()(x)
#
#     # Encoding layers
#     skip_connections = []
#     for i in range(num_layers):
#         x = _conv2d_block(inputs=x,
#                           num_filters=num_filters,
#                           use_batch_normalization=True,
#                           dropout_rate=0.0)
#         skip_connections.append(x)
#
#         x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
#
#         num_filters = num_filters * 2
#
#     # Bottleneck
#     # x = Dropout(dropout_rate)(x)
#     x = _conv2d_block(inputs=x,
#                       num_filters=num_filters,
#                       use_batch_normalization=True,
#                       dropout_rate=0.0)
#
#     # Decoding layers
#     for skip_connection in reversed(skip_connections):
#         num_filters = num_filters // 2
#
#         x = Conv2DTranspose(filters=num_filters,
#                             kernel_size=(2, 2),
#                             strides=(2, 2),
#                             padding='same')(x)
#
#         height_crop, width_crop = _get_cropping_dimensions(
#             target_tensor=skip_connection,
#             reference_tensor=x
#         )
#
#         if height_crop is not None and width_crop is not None:
#             skip_connection = Cropping2D(
#                 cropping=(height_crop, width_crop)
#             )(skip_connection)
#
#         x = concatenate([x, skip_connection])
#         x = _conv2d_block(inputs=x,
#                           num_filters=num_filters,
#                           use_batch_normalization=True,
#                           dropout_rate=0.0)
#
#     outputs = Conv2D(filters=num_classes,
#                      kernel_size=(1, 1),
#                      activation=output_activation)(x)
#
#     model = Model(inputs=[inputs], outputs=[outputs])
#
#     return model
