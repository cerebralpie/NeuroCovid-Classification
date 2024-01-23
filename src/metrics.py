"""

"""
import tensorflow as tf
from tensorflow.keras import layers


def continuous_dice_coefficient(y_true, y_pred):
    """
    Calculates the Continuous Dice Coefficient (CDC) between a ground truth
    (y_true) and a segmented image (y_pred).

    This function first flattens the input tensors, and then calculates the
    CDC based on the formula:
        CDC = 2 * |G ∩ S| / (c * |G| + |S|),
    where G is the ground truth, S is the 
    """
    y_true = layers.Flatten()(y_true)
    y_pred = layers.Flatten()(y_pred)

    sign_of_s = tf.sign(y_pred)
    size_of_g_intersect_s = tf.reduce_sum(y_true * y_pred)
    size_of_g = tf.reduce_sum(y_true)
    size_of_s = tf.reduce_sum(y_pred)

    if size_of_g_intersect_s > 0:
        c = size_of_g_intersect_s / tf.reduce_sum(y_true * sign_of_s)
    else:
        c = 1

    cdc_numerator = 2 * size_of_g_intersect_s
    cdc_denominator = (c * size_of_g) + size_of_s
    cdc = cdc_numerator / cdc_denominator

    return cdc


def continuous_dice_coefficient_loss(y_true, y_pred):
    return 1.0 - continuous_dice_coefficient(y_true, y_pred)

# Corrigir
def balanced_average_hausdorff_distance(y_true, y_pred):

    y_true = layers.Flatten()(y_true)
    y_pred = layers.Flatten()(y_pred)

    # pairwise distances between y_true and y_pred
    pairwise_distances = tf.norm(tf.expand_dims(y_true, axis=1) -
                                 tf.expand_dims(y_pred, axis=0), axis=-1)

    # compute minimum distances for each point in y_true to a point in y_pred
    min_distances_y_true = tf.reduce_min(pairwise_distances, axis=1)

    # compute minimum distances for each point in y_pred to a point in y_true
    min_distances_y_pred = tf.reduce_min(pairwise_distances, axis=0)

    num_pixels_y_true = tf.cast(tf.reduce_sum(y_true), tf.float32)

    # compute the balanced average Hausdorff distance
    bahd = ((tf.reduce_sum(min_distances_y_true) / num_pixels_y_true) +
            (tf.reduce_sum(min_distances_y_pred) / num_pixels_y_true)) / 2.0

    return bahd


def average_surface_distance(y_true, y_pred):
    pass
