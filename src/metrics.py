"""

"""
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    generate_binary_structure,
    binary_erosion
    )
from src.external.surface_distance.metrics import (
    compute_surface_distances,
    compute_average_surface_distance
)

# Allows tensor conversions to numpy
tf.config.run_functions_eagerly(True)


# Sørensen–Dice coefficient
def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
        For a binary segmentation task, calculate the Dice Coefficient (also
        known as F1-Score) between a ground truth G (y_true) and a predicted
        segmentation S (y_pred).

        This function first flattens the input tensors, and then calculates the
        Dice Coefficient based on the formula:
            Dice = (2 * |G ∩ S|) / (|G| + |S|),
        where G is the ground truth, and S is the segmented image.

        Args:
            y_true: A tensor containing the ground truth values.
            y_pred: A tensor containing the predicted values.
            smooth: A smoothing factor to prevent divisions by zero. Defaults
                    to 1.0.

        Returns:
            A tensor containing the Dice Coefficient with a value ranging from
            0 to 1. A Dice value of 1 indicates a perfect overlap, meaning the
            segmentation is identical to the ground truth. Conversely, a Dice
            value of 0 signifies no overlap between the segmented image and the
            ground truth.
        """
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    # Special Case 1: If both are all zeros
    if tf.reduce_all(tf.equal(y_true_flat, 0.0)) and tf.reduce_all(
            tf.equal(y_pred_flat, 0.0)):
        return tf.constant(1.0, dtype=tf.float32)

    # Special Case 2: Any non-zero in y_true_flat, but y_pred_flat is all zeros
    if tf.reduce_any(tf.not_equal(y_true_flat, 0.0)) and tf.reduce_all(
            tf.equal(y_pred_flat, 0.0)):
        return tf.constant(0.0, dtype=tf.float32)

    # Standard Dice Coefficient calculation (handles remaining cases)
    intersection = K.sum(y_true_flat * y_pred_flat)

    dice_numerator = 2.0 * intersection + smooth
    dice_denominator = K.sum(y_true) + K.sum(y_pred) + smooth

    dice_value = dice_numerator / dice_denominator

    return dice_value


# Dice Coefficient Loss
def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the Dice Coefficient Loss (also known as the Sørensen–Dice Loss)
    for binary segmentation tasks.

    This loss function is based on the Dice Coefficient, a measure of overlap
    between two sets. It's particularly useful for imbalanced segmentation problems.
    A lower Dice Coefficient Loss indicates a better overlap between the predicted
    segmentation and the ground truth.

    Args:
        y_true: A tensor containing the ground truth values.
        y_pred: A tensor containing the predicted values.

    Returns:
        A tensor containing the Dice Coefficient Loss, with a value ranging
        from 0 to 1. A loss of 0 indicates perfect segmentation, while a loss of
        1 indicates no overlap between the prediction and the ground truth.
    """
    return 1.0 - dice_coefficient(y_true=y_true, y_pred=y_pred)


# Continuous Dice Coefficient (CDC)
def continuous_dice_coefficient(y_true: tf.Tensor,
                                y_pred: tf.Tensor,
                                smooth=1.0) -> tf.Tensor:
    """
    For a binary segmentation task, calculate the Continuous Dice Coefficient
    (CDC) between a ground truth G (y_true) and a predicted segmentation S
    (y_pred).

    This function first flattens the input tensors, and then calculates the
    CDC based on the formula:
        CDC = 2 * |G ∩ S| / (c * |G| + |S|),
    where G is the ground truth, S is the segmented image, and c is a
    correction factor that depends on the size of the intersection of G and S.

    Args:
        y_true: A tensor containing the ground truth values.
        y_pred: A tensor containing the predicted values.
        smooth: A smoothing factor to prevent divisions by zero. Defaults to 1.

    Returns:
        A tensor containing the Continuous Dice Coefficient (CDC) with a value
        ranging from 0 to 1. A CDC value of 1 indicates a perfect overlap,
        meaning the segmentation is identical to the ground truth. Conversely,
        a CDC value of 0 signifies no overlap between the segmented image and
        the ground truth.
    """
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    y_true_flat = tf.cast(y_true_flat, tf.float32)
    y_pred_flat = tf.cast(y_pred_flat, tf.float32)

    # Special Case 1: If both are all zeros
    if tf.reduce_all(tf.equal(y_true_flat, 0.0)) and tf.reduce_all(
            tf.equal(y_pred_flat, 0.0)):
        return tf.constant(1.0, dtype=tf.float32)

    # Special Case 2: Any non-zero in y_true_flat, but y_pred_flat is all zeros
    if tf.reduce_any(tf.not_equal(y_true_flat, 0.0)) and tf.reduce_all(
            tf.equal(y_pred_flat, 0.0)):
        return tf.constant(0.0, dtype=tf.float32)

    # Standard CDC calculation (handles remaining cases)
    sign_of_s = K.sign(y_pred_flat)
    size_of_g_intersect_s = K.sum(y_true_flat * y_pred_flat)
    size_of_g = K.sum(y_true_flat)
    size_of_s = K.sum(y_pred_flat)

    c_numerator = size_of_g_intersect_s
    c_denominator = K.sum(y_true_flat * sign_of_s)

    c = tf.cond(
        pred=K.equal(c_denominator,
                     tf.cast(tf.constant(0.0), c_denominator.dtype)),
        true_fn=lambda: tf.constant(1.0, dtype=tf.float32),
        false_fn=lambda: (c_numerator / c_denominator)
    )

    cdc_numerator = 2.0 * size_of_g_intersect_s
    cdc_denominator = (c * size_of_g) + size_of_s
    cdc_value = (cdc_numerator + smooth) / (cdc_denominator + smooth)

    return cdc_value


# Continuous Dice Coefficient Loss (CDC loss)
def cdc_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Calculate the loss based on the Continuous Dice Coefficient (CDC).

    This function computes the loss as 1 - CDC, where CDC is a measure of the
    overlap between a predicted segmentation and its ground truth. A lower loss
    indicates better segmentation performance.

    Args:
        y_true: A tensor containing the ground truth values.
        y_pred: A tensor containing the predicted values.

    Returns:
        A tensor containing the CDC loss, with a value ranging from 0 to 1. A
        loss of 0 indicates perfect segmentation (complete overlap), while a
        loss of 1 means that there's no overlap between the predicted
        segmentation and the ground truth.
    """
    return 1.0 - continuous_dice_coefficient(y_true=y_true, y_pred=y_pred)


# Balanced Average Hausdorff Distance (BAHD)
def bahd(y_true: tf.Tensor, y_pred: tf.Tensor, smooth=1.0) -> tf.Tensor:
    """
    For a binary segmentation task, calculate the Balanced Average Hausdorff
    Distance (BAHD) between a ground truth G (y_true) and a predicted
    segmentation S (y_pred).

    This function first identifies the coordinates of points in the input
    tensors that belong to a region of interest (ROI), and then calculates the
    BAHD based on the formula:
        BAHD = (sum_min_distances_G_to_S / size_of_G ) +
               (sum_min_distances_S_to_G / size_of_G),
    where G is the ground truth, S is the segmented image,
    and sum_min_distances* represent the sum of the minimum distances from each
    point in one set to the other set.

    Args:
        y_true: A tensor containing the ground truth values.
        y_pred: A tensor containing the predicted values.
        smooth: A smoothing factor to prevent divisions by zero. Defaults to 1.

    Returns:
        A tensor containing the Balanced Average Hausdorff Distance (BAHD). The
        lower the BAHD value, the smaller is the distance between the ground
        truth and the segmented image, and the more similar their sizes are.
    """
    def _sum_min_distances(set_a: tf.Tensor, set_b: tf.Tensor) -> tf.Tensor:
        """
        Calculate the minimum distances between two sets of points.

        Args:
            set_a: A tensor of shape (n, d) representing n points in d
                  dimensions.
            set_b: A tensor of shape (m, d) representing m points in d
                  dimensions.

        Returns:
            A tensor of shape (n,) containing the minimum squared Euclidean
            distance from each point in set1 to any point in set2.
        """
        # shape.compare_batch_dimensions(
        #     tensors=(set_a, set_b),
        #     tensor_names=("set1", "set2"),
        #     last_axes=-3,
        #     broadcast_compatible=True)
        #
        # # Verify that the last axis of the tensors have the same dimension
        # dimension = set_a.shape.as_list()[-1]
        # shape.check_static(
        #     tensor=set_b,
        #     tensor_name="set2",
        #     has_dim_equals=(-1, dimension))

        # Create N x M matrix where the entry i,j corresponds to ai - bj
        # (vector of dimension D)
        result = (tf.expand_dims(set_a, axis=-2) -
                  tf.expand_dims(set_b, axis=-3))

        # Calculate the square distances between each two points: |ai - bj|^2.
        square_distances = tf.einsum("...i,...i->...", result, result)

        # Calculate the minimum square distances between a and b
        result = tf.reduce_min(
            input_tensor=square_distances, axis=-1)

        result = tf.reduce_sum(input_tensor=tf.sqrt(result), axis=-1)

        return result

    # Special Case 1: If both are all zeros
    if tf.reduce_all(tf.equal(y_true, 0.0)) and tf.reduce_all(
            tf.equal(y_pred, 0.0)):
        return tf.constant(0.0, dtype=tf.float32)

    # Special Case 2: Any non-zero in y_true_flat, but y_pred_flat is all zeros
    if tf.reduce_any(tf.not_equal(y_true, 0.0)) and tf.reduce_all(
            tf.equal(y_pred, 0.0)):
        return tf.constant(float('inf'), dtype=tf.float32)

    # Standard BAHD calculation (handles remaining cases)
    g_coords = tf.cast(tf.where(y_true > 0.5), dtype=tf.float16)
    s_coords = tf.cast(tf.where(y_pred > 0.5), dtype=tf.float16)

    size_of_g = tf.cast(tf.reduce_sum(y_true), dtype=tf.float16)

    sum_min_distances_g_to_s = _sum_min_distances(g_coords, s_coords)
    sum_min_distances_s_to_g = _sum_min_distances(s_coords, g_coords)

    bahd_left = (sum_min_distances_g_to_s + smooth) / (size_of_g + smooth)
    bahd_right = (sum_min_distances_s_to_g + smooth) / (size_of_g + smooth)

    bahd_value = tf.cast(bahd_left + bahd_right, dtype=tf.float32)

    return bahd_value


def avg_surface_distance(tensor1: tf.Tensor,
                         tensor2: tf.Tensor):
    input1 = tensor1.numpy()
    input2 = tensor2.numpy()
    input_1 = input1.astype(np.bool_)
    input_2 = input2.astype(np.bool_)
    input_1_reshaped = input_1.reshape(4, 384, 384)
    input_2__reshaped = input_2.reshape(4, 384, 384)
    distances_dict = compute_surface_distances(input1, input2, spacing_mm=(1, 1, 1))
    avg_distance = compute_average_surface_distance(distances_dict)

    return avg_distance

# def surface_distance_metric(tensor1: tf.Tensor,
#                             tensor2: tf.Tensor,
#                             sampling=1,
#                             connectivity=1):
#     return 1
#     # input1 = tensor1.numpy()
#     # input2 = tensor2.numpy()
#     # input_1 = np.atleast_1d(input1.astype(np.bool_))
#     # input_2 = np.atleast_1d(input2.astype(np.bool_))
#     #
#     # conn = generate_binary_structure(input_1.ndim, connectivity)
#     #
#     # s = input_1 ^ binary_erosion(input_1, conn)
#     # sprime = input_2 ^ binary_erosion(input_2, conn)
#     #
#     # dta = distance_transform_edt(~s, sampling)
#     # dtb = distance_transform_edt(~sprime, sampling)
#     #
#     # sds = np.concatenate([np.ravel(dta[sprime != 0]), np.ravel(dtb[s != 0])])
#     # avg_distance = sds.mean()
#     #
#     # return avg_distance
