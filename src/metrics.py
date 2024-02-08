"""

"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_graphics.util import shape


# Continuous Dice Coefficient (CDC)
def cdc(y_true: tf.Tensor, y_pred: tf.Tensor, smooth=1) -> tf.Tensor:
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
    y_true = layers.Flatten()(y_true)
    y_pred = layers.Flatten()(y_pred)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sign_of_s = tf.sign(y_pred)
    size_of_g_intersect_s = tf.reduce_sum(y_true * y_pred)
    size_of_g = tf.reduce_sum(y_true)
    size_of_s = tf.reduce_sum(y_pred)

    c_numerator = size_of_g_intersect_s
    c_denominator = tf.reduce_sum(y_true * sign_of_s)

    c = tf.cond(
        pred=tf.equal(c_denominator,
                      tf.cast(tf.constant(0.0), c_denominator.dtype)),
        true_fn=lambda: tf.constant(1.0, dtype=tf.float32),
        false_fn=lambda: (c_numerator / c_denominator)
    )

    cdc_numerator = 2 * size_of_g_intersect_s
    cdc_denominator = (c * size_of_g) + size_of_s
    cdc_value = ((cdc_numerator + smooth) / (cdc_denominator + smooth))

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
    return 1.0 - cdc(y_true=y_true, y_pred=y_pred)


# Balanced Average Hausdorff Distance (BAHD)
def bahd(y_true: tf.Tensor, y_pred: tf.Tensor, smooth=1) -> tf.Tensor:
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
        shape.compare_batch_dimensions(
            tensors=(set_a, set_b),
            tensor_names=("set1", "set2"),
            last_axes=-3,
            broadcast_compatible=True)

        # Verify that the last axis of the tensors have the same dimension
        dimension = set_a.shape.as_list()[-1]
        shape.check_static(
            tensor=set_b,
            tensor_name="set2",
            has_dim_equals=(-1, dimension))

        # Create N x M matrix where the entry i,j corresponds to ai - bj
        # (vector of dimension D)
        difference = (
                tf.expand_dims(set_a, axis=-2) -
                tf.expand_dims(set_b, axis=-3))

        # Calculate the square distances between each two points: |ai - bj|^2.
        square_distances = tf.einsum("...i,...i->...", difference, difference)

        minimum_square_distance_a_to_b = tf.reduce_min(
            input_tensor=square_distances, axis=-1)

        return tf.sqrt(
            tf.reduce_sum(input_tensor=minimum_square_distance_a_to_b, axis=-1))

    g_coords = tf.where(y_true > 0.5)
    s_coords = tf.where(y_pred > 0.5)

    size_of_g = tf.constant(g_coords.shape.as_list()[0], dtype=tf.float32)

    sum_min_distances_g_to_s = _sum_min_distances(g_coords, s_coords)
    sum_min_distances_s_to_g = _sum_min_distances(s_coords, g_coords)

    sum_min_distances_g_to_s = tf.reduce_sum(sum_min_distances_g_to_s)
    sum_min_distances_s_to_g = tf.reduce_sum(sum_min_distances_s_to_g)

    bahd_left = (sum_min_distances_g_to_s + smooth) / (size_of_g + smooth)
    bahd_right = (sum_min_distances_s_to_g + smooth) / (size_of_g + smooth)

    return bahd_left + bahd_right


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
