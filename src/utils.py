"""
File's docstring - To be done
"""
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import pydicom
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split


SEED = 42


def configure_gpu() -> tf.distribute.Strategy:
    """
    Configures the GPU settings for TensorFlow if a GPU is available.

    This function lists all the physical GPU devices available and sets the
    memory growth of each GPU. It also sets the first GPU device as the visible
    device for TensorFlow and creates a device strategy fot it.

    If no GPU is found, it gets the default distribution strategy and prints
    the number of replicas in sync.

    Returns:
        A TensorFlow device strategy

    Raises:
        RuntimeError: If there is an error while setting the GPU settings.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            print("\nGPU Found! Using GPU...")
        except RuntimeError:
            raise
    else:
        strategy = tf.distribute.get_strategy()

    return strategy


def dicom_to_tensor(dicom_file_path: Path) -> tf.Tensor:
    file = pydicom.dcmread(fp=dicom_file_path)
    image_array = file.pixel_array

    tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

    return tensor


def load_data(data_root_directory: Path,
              split: float = 0.1) -> tuple[tuple[Path, Path],
                                           tuple[Path, Path],
                                           tuple[Path, Path]]:
    """
    Load and split image and mask paths from a given directory for machine
    learning tasks.

    This function searches for image and mask data within the specified
    'data_root_directory' and loads their paths. The data is then split into
    training, validation, and test sets using the provided 'split' ratio.

    Args:
        data_root_directory: pathlib.Path object to the root directory
                             containing "images" and "masks" subdirectories.
        split: Fraction of data to allocate for validation and test sets.
               Defaults to 0.1.

    Returns:
        Returns a tuple containing three sub-tuples of image and mask paths:
            - (training_image_paths, training_mask_paths)
            - (validation_image_paths, validation_mask_paths)
            - (test_image_paths, test_mask_paths)

    Raises:
        FileNotFoundError:
            - If the 'root_dir' or its "images" or "masks" subdirectories are
              not found.
            - If there are no image or mask files within the respective
              subdirectories.
        ValueError: If the number of images and masks does not match.
    """
    try:
        image_paths = sorted(data_root_directory.glob("images/*"))
        mask_paths = sorted(data_root_directory.glob("masks/*"))

        if not image_paths or not mask_paths:
            raise FileNotFoundError("Images or masks not found in the "
                                    "specified directories.")

        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks does not match.")

        total_image_mask_pairs = len(image_paths)
        split_sizes = int(total_image_mask_pairs * split)

        x_train, x_valid = train_test_split(image_paths,
                                            test_size=split_sizes,
                                            random_state=SEED)
        y_train, y_valid = train_test_split(mask_paths,
                                            test_size=split_sizes,
                                            random_state=SEED)

        x_train, x_test = train_test_split(x_train,
                                           test_size=split_sizes,
                                           random_state=SEED)
        y_train, y_test = train_test_split(y_train,
                                           test_size=split_sizes,
                                           random_state=SEED)

        training_paths = (x_train, y_train)
        validation_paths = (x_valid, y_valid)
        test_paths = (x_test, y_test)

        return training_paths, validation_paths, test_paths

    except (FileNotFoundError, ValueError):
        raise


def read_image(image_path: Path,
               image_shape: tuple[int, int],
               grayscale: bool = False) -> np.ndarray:
    """
    Read, resize, and preprocess an image from a given path.

    This function handles both standard image formats (e.g., PNG, JPEG) and
    DICOM files. it resizes the image to the specified shape and normalizes
    pixel values to the range [0, 1]

    Args:
        image_path: Path to the image file.
        image_shape: Desired shape (width and height) of the resized image.
        grayscale: Whether to convert the image to grayscale instead of RGB.
                   Defaults to False.

    Returns:
        A NumPy array representing the preprocessed image. The array is of
        shape (height, width, 3) for RGB images and (height, width, 1) for
        grayscale images. Pixel values are normalized to [0, 1].

    Raises:
        FileNotFoundError: If the image file cannot be found at the given path.
        cv2.error: If an error occurs during image reading or processing with
                   OpenCV.
    """
    try:
        # Check if the image is a DICOM file
        if image_path.suffix.lower() in [".dcm", ".dicom"]:
            image_bytes = tf.io.read_file(image_path)
            image_tensor = tfio.image.decode_dicom_image(image_bytes,
                                                         dtype=tf.uint8,
                                                         color_dim=True,
                                                         on_error='lossy')
            raw_image_array = image_tensor.numpy()
        else:
            path = str(image_path)
            color_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            raw_image_array = cv2.imread(path, color_mode)

            if raw_image_array is None:
                raise FileNotFoundError(f"Image not found at path: {path}")

            if not grayscale:
                raw_image_array = cv2.cvtColor(raw_image_array,
                                               cv2.COLOR_BGR2RGB)

        # Resize the image
        resized_image_array = cv2.resize(raw_image_array, image_shape)

        # Normalize the pixel values to the range [0, 1]
        normalized_image_array = resized_image_array / np.linalg.norm(
            resized_image_array)

        if grayscale:
            normalized_image_array = np.expand_dims(normalized_image_array,
                                                    axis=-1)

        return normalized_image_array

    except (FileNotFoundError, cv2.error):
        raise


def get_tensorflow_dataset(image_mask_paths: tuple[Path, Path],
                           image_size: int,
                           batch_size: int = 32):
    def _parse(image_path: Path, mask_path: Path):
        def _subparse(sub_image_path: Path, sub_mask_path: Path):
            image_array = read_image(image_path=sub_image_path,
                                     grayscale=False)
            mask_array = read_image(image_path=sub_mask_path,
                                    grayscale=True)

            return image_array, mask_array

        image_tensor, mask_tensor = tf.numpy_function(_subparse,
                                                      [image_path, mask_path],
                                                      [tf.float32, tf.float32])
        image_tensor.set_shape([image_size, image_size, 3])
        mask_tensor.set_shape([image_size, image_size, 1])

        return image_tensor, mask_tensor

    dataset = tf.data.Dataset.from_tensor_slices((image_mask_paths[0],
                                                 image_mask_paths[1]))
    dataset = dataset.map(_parse)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat()

    return dataset
