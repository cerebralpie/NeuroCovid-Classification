"""
File's docstring - To be done
"""
import tensorflow as tf
import pydicom
import cv2
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

    return strategy


def dicom_to_tensor(dicom_file_path: Path) -> tf.Tensor:
    file = pydicom.dcmread(fp=dicom_file_path)
    image_array = file.pixel_array

    tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

    return tensor


def load_data(root_dir: Path, split=0.1):
    """
    Load and split image and mask paths from a given directory for machine
    learning tasks.

    This function searches for image and mask data within the specified
    'root_dir'. It loads their paths, ensuring consistency and raising errors
    for potential issues. The data is then split into training, validation, and
    test sets using the provided 'split' ratio.

    Args:
        root_dir: pathlib.Path object to the root directory containing
                  "images" and "masks" subdirectories.
        split: Fraction of data to allocate for validation and test sets
               (default 0.1 each)

    Returns:
        Returns None if an error occurs during data loading, validation,
        or splitting. Otherwise, returns a tuple containing three sub-tuples of
        image and mask paths:
            - (training_images, training_masks)
            - (validation_images, validation_masks)
            - (test_images, test_masks)

    Raises:
        FileNotFoundError:
            - If the 'root_dir' or its "images" or "masks" subdirectories are
              not found.
            - If there are no image or mask files within the respective
              subdirectories.
        ValueError: If the number of images and masks does not match.
    """
    try:
        images = sorted(root_dir.glob("images/*"))
        masks = sorted(root_dir.glob("masks/*"))

        if not images or not masks:
            raise FileNotFoundError("Images or masks not found in the "
                                    "specified directories.")

        if len(images) != len(masks):
            raise ValueError("Number of images and masks does not match.")

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

    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        return None


def read_image(path: Path, image_shape: tuple[int, int]):
    """
    Read, resize, and preprocess an image from a given path.

    This function read an image from the specified 'path', resizes it to the
    desired 'image_shape', and normalizes pixel values to the range [0, 1].

    Args:
        path: pathlib.Path object to the image file.
        image_shape: Desired shape (width and height) of the resized image.

    Returns:
        A NumPy array representing the preprocessed image, or None if an error
        occurs during reading or processing.

    Raises:
        FileNotFoundError: If the image file cannot be found at the given path.
        cv2.error: If an error occurs during image reading or processing with
                   OpenCV.
    """
    try:
        path = str(path)
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Image not found at path: {path}")

        image = cv2.resize(image, image_shape)
        image = image / 255.0

        return image

    except (FileNotFoundError, cv2.error) as e:
        print(f"Error reading image: {e}")
        return None
