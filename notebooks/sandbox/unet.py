import os.path

import numpy as np
import tensorflow as tf
import src.metrics as nc_metrics
import src.utils as nc_utils
import notebooks.sandbox.models as nc_models

from pathlib import Path
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import models

from tensorflow.keras.metrics import Recall, Precision


if __name__ == '__main__':
    strategy = nc_utils.start_session()

    ROOT_DIR = "."
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    MASK_DIR = os.path.join(ROOT_DIR, "masks")

    BATCH_SIZE = 8
    LR = 1e-4  # Learning rate
    EPOCHS = 300

    smallest_dimension = nc_utils.get_smallest_image_dimension(IMAGE_DIR)

    #IMAGE_SIZE = smallest_dimension
    IMAGE_SIZE = 256
    IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
    INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

    (x_train_paths, y_train_paths), (x_valid_paths, y_valid_paths), (
    x_test_paths,
    y_test_paths) = nc_utils.load_data(
        image_directory=IMAGE_DIR,
        mask_directory=MASK_DIR,
        split=0.1
    )

    train_dataset = nc_utils.get_tensorflow_dataset(
        image_mask_paths=(x_train_paths, y_train_paths),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_dataset = nc_utils.get_tensorflow_dataset(
        image_mask_paths=(x_valid_paths, y_valid_paths),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    test_dataset = nc_utils.get_tensorflow_dataset(
        image_mask_paths=(x_test_paths, y_test_paths),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    data_aug = nc_utils.get_data_augmentation_pipeline()

    # Initiating model on GPU
    with strategy.scope():
        model = nc_models.unet_model(input_shape=INPUT_SHAPE, augment_data=True, num_filters=8)
        metrics=[Recall(), Precision()]
        loss=nc_metrics.cdc_loss
        opt=tf.keras.optimizers.Nadam(LR)

    # Compiling model
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metrics)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                             patience=5, mode='max',
                                             restore_best_weights=True)

    # Model Checkpoint

    train_steps = len(x_train_paths) // BATCH_SIZE
    valid_steps = len(x_valid_paths) // BATCH_SIZE

    if len(x_train_paths) % BATCH_SIZE != 0:
        train_steps += 1
    if len(x_valid_paths) % BATCH_SIZE != 0:
        valid_steps += 1

    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=validation_dataset,
            callbacks=[early_stopping],
            steps_per_epoch=train_steps,
            validation_steps=valid_steps)
    except Exception as e:
        print("An error occurred:", e)

