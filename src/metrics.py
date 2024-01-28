"""

"""
import tensorflow as tf
from tensorflow.keras import layers


def cdc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    For a binary segmentation task, calculate the Continuous Dice Coefficient
    given a ground truth G (y_true) and a predicted segmentation S (y_pred).

    This function first flattens the input tensors, and then calculates the
    CDC based on the formula:
        CDC = 2 * |G ∩ S| / (c * |G| + |S|),
    where G is the ground truth, S is the segmented image, and c is a
    correction factor that depends on the size of the intersection of G and S.

    Args:
        y_true (tf.Tensor): The ground truth values
        y_pred (tf.Tensor): The predicted values

    Returns:
        The Continuous Dice Coefficient (CDC) with a value ranging from 0 to 1.
        A CDC value of 1 indicates a perfect overlap, meaning the segmentation
        is identical to the ground truth. Conversely, a cDC value of 0 signifies
        no overlap between the segmented image and the ground truth.
    """
    y_true = layers.Flatten()(y_true)
    y_pred = layers.Flatten()(y_pred)

    sign_of_s = tf.sign(y_pred)
    size_of_g_intersect_s = tf.reduce_sum(y_true * y_pred)
    size_of_g = tf.reduce_sum(y_true)
    size_of_s = tf.reduce_sum(y_pred)

    if tf.equal(size_of_g, 0.0).numpy():
        if tf.equal(size_of_s, 0.0).numpy():
            cdc_value = tf.constant(float(1.0), dtype=tf.float32)
        else:
            cdc_value = tf.constant(float(0.0), dtype=tf.float32)
    elif tf.equal(size_of_s, 0.0).numpy():
        cdc_value = tf.constant(float(0.0), dtype=tf.float32)
    else:
        if tf.greater(size_of_g_intersect_s, 0.0).numpy():
            c = size_of_g_intersect_s / (tf.reduce_sum(y_true * sign_of_s) +
                                         1e-7)
        else:
            c = 1.0

        cdc_numerator = 2 * size_of_g_intersect_s
        cdc_denominator = (c * size_of_g) + size_of_s
        cdc_value = (cdc_numerator / (cdc_denominator + 1e-7))

    return cdc_value


def cdc_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Calculates the loss based on the Continuous Dice Coefficient (CDC).

    This function computes the loss as 1 - CDC, where CDC is a measure of overlap
    between a predicted segmentation and its ground truth. A lower loss indicates
    better segmentation performance.

    Args:
        y_true (tf.Tensor): The ground truth values
        y_pred (tf.Tensor): The predicted values

    Returns:
        The CDC loss, with a value ranging from 0 to 1.
        A loss of 0 indicates perfect segmentation (complete overlap), while a loss
        of 1 signifies no overlap between the predicted segmentation and ground truth.
    """
    return 1.0 - cdc(y_true=y_true, y_pred=y_pred)


def bahd(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

    g_coords = tf.where(y_true == 1)
    s_coords = tf.where(y_pred == 1)

    size_of_g = tf.shape(g_coords)[0]
    size_of_s = tf.shape(s_coords)[0]

    size_of_g = tf.constant(size_of_g, dtype=tf.int8)
    size_of_s = tf.constant(size_of_s, dtype=tf.int8)

    if tf.equal(size_of_g, 0).numpy():
        if tf.equal(size_of_s, 0).numpy():
            bahd_value = tf.constant(float(0.0), dtype=tf.float32)
        else:
            bahd_value = tf.constant(float('inf'), dtype=tf.float32)
    elif tf.equal(size_of_s, 0).numpy():
        bahd_value = tf.constant(float('inf'), dtype=tf.float32)
    else:
        # Expand dimensions for pairwise operations
        g_coords_expanded = tf.expand_dims(g_coords, axis=1)
        s_coords_expanded = tf.expand_dims(s_coords, axis=0)

        g_coords_expanded = tf.cast(g_coords_expanded, tf.float32)
        s_coords_expanded = tf.cast(s_coords_expanded, tf.float32)

        g_coords_expanded_swapped = tf.expand_dims(g_coords, axis=0)
        s_coords_expanded_swapped = tf.expand_dims(s_coords, axis=1)

        g_coords_expanded_swapped = tf.cast(g_coords_expanded_swapped,
                                            tf.float32)
        s_coords_expanded_swapped = tf.cast(s_coords_expanded_swapped,
                                            tf.float32)

        # Calculate pairwise distances
        distances_y_true = tf.norm(g_coords_expanded - s_coords_expanded,
                                   axis=2)
        distances_y_pred = tf.norm(g_coords_expanded_swapped -
                                   s_coords_expanded_swapped, axis=2)

        # Find minimum distances (optional)
        min_distances_y_true = tf.reduce_min(distances_y_true, axis=1)
        min_distances_y_pred = tf.reduce_min(distances_y_pred, axis=1)

        size_of_g = tf.cast(size_of_g, tf.float32)

        bahd_value = (tf.reduce_sum(min_distances_y_true) /
                      (size_of_g + 1e-7) +
                      tf.reduce_sum(min_distances_y_pred) /
                      (size_of_g + 1e-7)) / 2.0

    return bahd_value


def surface_distance_metric(y_true, y_pred, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Calculate the average surface distance between two segmentation masks.

    Args:
        y_true: Ground truth segmentation mask.
        y_pred: Predicted segmentation mask.
        voxel_spacing: Tuple specifying the voxel spacing in each dimension
        (x, y, z).
    Returns:
        The Average surface distance.
    """
    # Ensure tensors have the same shape
    tf.debugging.assert_shapes([(y_true, ('B', 'H', 'W', 'D')),
                                (y_pred, ('B', 'H', 'W', 'D'))])

    # Flatten the masks
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # Find the surface voxels
    true_surface = tf.cast(tf.equal(y_true_flat, 1), tf.float32)
    pred_surface = tf.cast(tf.equal(y_pred_flat, 1), tf.float32)

    # Compute distance transform
    true_distance = tf.nn.morphology.distance_transform_edt(tf.cast(true_surface, tf.uint8), voxel_spacing)
    pred_distance = tf.nn.morphology.distance_transform_edt(tf.cast(pred_surface, tf.uint8), voxel_spacing)

    # Compute average surface distance
    avg_surface_distance = tf.reduce_sum(tf.abs(true_distance - pred_distance)) / tf.reduce_sum(true_surface)

    return avg_surface_distance
