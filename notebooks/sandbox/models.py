import tensorflow as tf

import src.utils as nc_utils
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Conv2DTranspose,
    MaxPooling2D, Dropout, Input, concatenate, Cropping2D
)


def _get_cropping_dimensions(
        target_tensor: tf.Tensor,
        reference_tensor: tf.Tensor) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Calculate the cropping dimensions needed to align a target tensor with a
    reference tensor.

    This function calculates the top and bottom (height) and left and right
    (width) cropping dimensions required to make the 'target' tensor have the
    same spatial dimensions (height, width) as the 'reference' tensor. It
    assumes that the depth (channels) and batch dimensions are already
    compatible.

    Args:
        target_tensor: The tensor to be cropped.
        reference_tensor: The reference tensor to align with.

    Returns:
        A tuple of two tuples, representing the cropping dimensions for height
        and width, respectively. Each inner tuple contains the starting and
        ending indices for cropping.

    Raises:
        ValueError: If the target tensor has smaller height or width than the
                    reference tensor.
    """
    target_shape = K.int_shape(target_tensor)
    reference_shape = K.int_shape(reference_tensor)

    target_height, target_width = target_shape[1], target_shape[2]
    reference_height, reference_width = reference_shape[1], reference_shape[2]

    total_height_crop = target_height - reference_height
    total_width_crop = target_width - reference_width

    if total_height_crop <= 0 and total_width_crop <= 0:
        raise ValueError("The target tensor has smaller height and width than "
                         "the reference tensor.")
    elif total_height_crop <= 0:
        raise ValueError("The target tensor has smaller height than the "
                         "reference tensor.")
    elif total_width_crop <= 0:
        raise ValueError("The target tensor has smaller width than the "
                         "reference tensor.")

    if total_height_crop % 2 == 0:
        top_crop, bottom_crop = total_height_crop // 2, total_height_crop // 2
    else:
        top_crop, bottom_crop = (total_height_crop // 2, (total_height_crop //
                                 2) + 1)

    if total_width_crop % 2 == 0:
        left_crop, right_crop = total_width_crop // 2, total_width_crop // 2
    else:
        left_crop, right_crop = total_width_crop // 2, total_width_crop // 2 + 1

    return (top_crop, bottom_crop), (left_crop, right_crop)
