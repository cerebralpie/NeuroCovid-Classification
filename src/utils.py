"""
File's docstring - To be done
"""
import tensorflow as tf
import pydicom
from pathlib import Path
from sklearn.model_selection import train_test_split


SEED = 42


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


def dicom_to_tensor(dicom_file_path: Path) -> tf.Tensor:
    file = pydicom.dcmread(fp=dicom_file_path)
    image_array = file.pixel_array

    tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

    return tensor


def load_data(root_dir: Path, split=0.1):
    images = sorted(root_dir.glob("images/*"))
    masks = sorted(root_dir.glob("masks/*"))

    total_size = len(images)
    valid_size = int(total_size * split)
    test_size = int(total_size * split)

    x_train, x_valid = train_test_split(images, test_size=valid_size,
                                        random_state=SEED)
    y_train, y_valid = train_test_split(masks, test_size=valid_size,
                                        random_state=SEED)

    x_train, x_test = train_test_split(x_train, test_size=test_size,
                                       random_state=SEED)
    y_train, y_test = train_test_split(y_train, test_size=test_size,
                                       random_state=SEED)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
