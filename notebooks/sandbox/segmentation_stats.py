import tensorflow as tf
from keras.layers import Flatten


def continuous_dice_coefficient(y_true, y_pred):

    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)

    intersect = (y_true * y_pred)

    c = tf.reduce_sum(intersect) / tf.maximum(tf.cast(tf.size(
        tf.boolean_mask(intersect, intersect > 0)), tf.float32), 1.0)

    cdc = (2 * tf.reduce_sum(intersect)) / (c * tf.reduce_sum(y_true) +
                                            tf.reduce_sum(y_pred))

    return cdc


def continuous_dice_coefficient_loss(y_true, y_pred):
    return 1.0 - continuous_dice_coefficient(y_true, y_pred)


def balanced_average_hausdorff_distance(y_true, y_pred):

    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)

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

