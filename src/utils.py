"""
File's docstring - To be done
"""
import tensorflow as tf


def configure_gpu():
    """
    Configures the GPU settings for TensorFlow if a GPU is available.

    This function lists all the physical GPU devices available and sets the
    memory growth of each GPU. It also sets the first GPU device as the visible
    device for TensorFlow and creates a device strategy fot it.

    If no GPU is found, it gets the default distribution strategy and prints
    the number of replicas in sync.

    Raises:
        RuntimeError: If there is an error while setting the GPU settings.

    Returns:
        None
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            print("\nGPU Found! Using GPU...")
        except RuntimeError as e:
            print(e)
    else:
        strategy = tf.distribute.get_strategy()
        print("\nNumber of replicas: ", strategy.num_replicas_in_sync)
