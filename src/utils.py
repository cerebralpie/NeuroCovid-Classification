"""
File's docstring - To be done
"""
import numpy as np
import os
import tensorflow as tf
import tensorflow_io as tfio
import cv2
from pathlib import Path
from PIL import Image
from PIL import UnidentifiedImageError
from sklearn.model_selection import train_test_split


SEED = 42


def start_session() -> tf.distribute.OneDeviceStrategy:
    """
    Configure the GPU settings for TensorFlow if a GPU is available.

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
    tf.keras.backend.clear_session()

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        except RuntimeError:
            raise
    else:
        strategy = tf.distribute.get_strategy()

    return strategy


# def dicom_to_tensor(dicom_file_path: Path) -> tf.Tensor:
#     file = pydicom.dcmread(fp=dicom_file_path)
#     image_array = file.pixel_array
#
#     tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
#
#     return tensor


def load_data(image_directory: str,
              mask_directory: str,
              split: float = 0.1) -> tuple[tuple[list[str], list[str]],
                                           tuple[list[str], list[str]],
                                           tuple[list[str], list[str]]]:
    """
    Load and split image and mask paths from a given directory for machine
    learning tasks.

    This function searches for image and mask data within the specified
    'data_root_directory' and loads their paths. The data is then split into
    training, validation, and test sets using the provided 'split' ratio.

    Args:
        image_directory: The path to the directory containing the images.
        mask_directory: The path to the directory containing the masks.
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
        image_paths = sorted(os.path.join(image_directory, filename)
                             for filename in os.listdir(image_directory))
        mask_paths = sorted(os.path.join(mask_directory, filename)
                            for filename in os.listdir(mask_directory))

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


def read_image(image_path: str,
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
        filename = os.path.basename(image_path)
        _, suffix = os.path.splitext(filename)

        # Check if the image is a DICOM file
        if suffix.lower() in [".dcm", ".dicom"]:
            image_bytes = tf.io.read_file(image_path)
            image_tensor = tfio.image.decode_dicom_image(image_bytes,
                                                         dtype=tf.uint8,
                                                         color_dim=True,
                                                         on_error='lossy')
            raw_image_array = image_tensor.numpy()
        else:
            color_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            raw_image_array = cv2.imread(image_path, color_mode)

        if raw_image_array is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        else:
            raw_image_array = raw_image_array.astype('f')

        if not grayscale:
            raw_image_array = cv2.cvtColor(raw_image_array,
                                           cv2.COLOR_BGR2RGB)

        # Resize the image
        resized_image_array = cv2.resize(raw_image_array, image_shape)

        # Normalize the pixel values to the range [0, 1]
        normalized_image_array = resized_image_array / 255.0

        if grayscale:
            normalized_image_array = np.expand_dims(normalized_image_array,
                                                    axis=-1)

        return normalized_image_array

    except (FileNotFoundError, cv2.error):
        raise


def get_tensorflow_dataset(image_mask_paths: tuple[list[str], list[str]],
                           image_size: int,
                           batch_size: int = 32) -> tf.data.Dataset:
    """
    Generate a TensorFlow dataset from given image and mask paths.

    This function reads the images and masks from the provided paths, resizes
    them to the specified size, and returns a batched TensorFlow dataset. The
    dataset repeats indefinitely.

    Args:
        image_mask_paths: A tuple containing two lists of pathlib.Path objects.
                          The first list contains the paths to the images and
                          the second list contains the paths to the masks.
        image_size: The size to which images and masks will be resized. The
                    images and masks are assumed to be square.
        batch_size: The number of elements in each batch of the dataset.
                    Defaults to 32.

    Returns:
        A batched TensorFlow dataset (tf.data.Dataset) containing the images and
        masks.

    Raises:
        FileNotFoundError: If an image or mask file does not exist.
        ValueError: If an image or mask cannot be read or resized.
    """
    def _parse_image_and_mask(
            image_path_tensor: tf.Tensor,
            mask_path_tensor: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Parse the image and mask paths and returns the corresponding tensors.

        Args:
            image_path_tensor: A tensor containing the path to a single image.
            mask_path_tensor: A tensor containing the path to the ground truth
                              of the image in image_path_tensor.

        Returns:
            A tuple containing the image tensor and mask tensor.
        """
        def _read_and_process_image_and_mask(
                image_path_bytes: bytes,
                mask_path_bytes: bytes
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            Read the image and mask files and returns the corresponding arrays.

            Args:
                image_path_bytes: Bytes corresponding to the path to the image
                mask_path_bytes: Bytes corresponding to the path to the mask.

            Returns:
                A tuple containing the image array and mask array
            """
            sub_image_path = image_path_bytes.decode('utf-8')
            sub_mask_path = mask_path_bytes.decode('utf-8')

            image_array = read_image(image_path=sub_image_path,
                                     image_shape=(image_size, image_size),
                                     grayscale=False)
            mask_array = read_image(image_path=sub_mask_path,
                                    image_shape=(image_size, image_size),
                                    grayscale=True)

            return image_array, mask_array

        image_tensor, mask_tensor = tf.numpy_function(
            _read_and_process_image_and_mask,
            [image_path_tensor, mask_path_tensor],
            [tf.float32, tf.float32]
        )
        image_tensor.set_shape([image_size, image_size, 3])
        mask_tensor.set_shape([image_size, image_size, 1])

        return image_tensor, mask_tensor

    dataset = tf.data.Dataset.from_tensor_slices((image_mask_paths[0],
                                                  image_mask_paths[1]))
    dataset = dataset.map(_parse_image_and_mask)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat()

    return dataset


def get_data_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Create a data augmentation pipeline for brain MRI images.

    This function defines a Keras Sequential model that performs random
    translation and rotation on input images. This augmentation pipeline is
    specifically designed for brain MRI data, using conservative values to
    avoid introducing unrealistic distortions.

    Returns:
        A Keras Sequential model representing the data augmentation pipeline.
    """
    augmentation_pipeline = tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(
                height_factor=(-0.05, 0.05),
                width_factor=(-0.05, 0.05),
                fill_mode='reflect',
                interpolation='bilinear',
                seed=SEED
            ),

            tf.keras.layers.RandomRotation(
                factor=(-0.05, 0.05),
                fill_mode='reflect',
                interpolation='bilinear',
                seed=SEED
            ),
        ]
    )

    return augmentation_pipeline


def get_unique_dimensions(image_path: str) -> set[tuple[int, int]]:
    """
    Get the unique dimensions of all images in a specified directory

    This function iterates over all files in the specified directory, opens each
    file as an image using the PIL library, and appends the size of the image
    (width, height) to a set. The set of unique dimensions is then returned.

    Args:
        image_path: The path to the directory containing the images.

    Returns:
        A set of tuples, where each tuple represents the unique (width, height)
        dimensions of the images in the directory.

    Raises:
        UnidentifiedImageException: An error occurred when the image file
                                    cannot be identified.
        IOError: An error occurred when an input/output operation fails,
                 such as the file not being readable.
    """
    unique_dimensions = set()

    for filename in os.listdir(image_path):
        filepath = os.path.join(image_path, filename)
        if os.path.isfile(filepath):
            try:
                with Image.open(filepath) as img:
                    unique_dimensions.add(img.size)
            except (UnidentifiedImageError, IOError):
                raise

    return unique_dimensions


def get_smallest_image_dimension(image_path: str) -> int:
    """
    Find the smallest height or width of all images in a directory.

    This function uses the 'get_unique_dimensions' function to retrieve unique
    (width, height) dimensions of all images in a directory. It then iterates
    through these dimensions and finds the smallest value (either height or
    width) across all images. Finally, it returns this smallest value as an
    integer.

    Args:
        image_path: The path to the directory containing the images.

    Returns:
        An integer representing the smallest height or width value found amongst
        all images in the directory.
    """
    unique_dimensions = get_unique_dimensions(image_path)

    smallest_dimension = float('inf')

    for width, height in unique_dimensions:
        smallest_dimension = min(smallest_dimension, min(width, height))

    return int(smallest_dimension)
