import gzip
import numpy as np
import struct
import sys


def load_mnist_ubyte(image_path):
    """
    Loads MNIST images from the raw ubyte files.

    Args:
        image_path (str): Path to the image file (e.g., 'train-images-idx3-ubyte.gz').

    Returns:
        images (numpy.ndarray)
    """
    with gzip.open(image_path, 'rb') as f:
        # Read the header: magic number (4 bytes) + num images (4 bytes) +
        # num rows (4 bytes) + num cols (4 bytes) = 16 bytes.

        # Read the entire file content into a buffer
        image_data = f.read()

        # The image data starts at byte 16.
        images = np.frombuffer(image_data, dtype=np.uint8, offset=16)

        # We need the dimensions to reshape. We can extract them from the header bytes,
        # which are big-endian ('>'). We use struct.unpack if we were being strict,
        # but here we'll assume the standard MNIST format and calculate the dimensions
        # for a clean numpy approach.

        # The number of images is in the 5th to 8th byte (4 bytes)
        num_images = np.frombuffer(image_data, dtype='>i4', offset=4, count=1)[0]
        # Rows and columns are 28x28 for MNIST, stored in bytes 9-12 and 13-16.
        # num_rows = np.frombuffer(image_data, dtype='>i4', offset=8, count=1)[0]
        # num_cols = np.frombuffer(image_data, dtype='>i4', offset=12, count=1)[0]
        num_rows = 28
        num_cols = 28

        # Reshape the 1D array into a 3D array (num_images, rows, columns)
        images = images.reshape(num_images, num_rows, num_cols)
    return images

def load_mnist_labels(label_path):
    """
    Loads MNIST labels from the raw ubyte files.

    Args:
        label_path (str): Path to the label file (e.g., 'train-labels-idx1-ubyte.gz').

    Returns:
        labels (numpy.ndarray)
    """
    with gzip.open(label_path, 'rb') as f:
        # Read the header: magic number (4 bytes) + num items (4 bytes) = 8 bytes.
        # Skip these 8 bytes.

        # Read the entire file content into a buffer
        label_data = f.read()

        # The label data starts at byte 8. The data type is unsigned byte ('B' or np.uint8).
        # Labels are a 1D vector.
        labels = np.frombuffer(label_data, dtype=np.uint8, offset=8)

    return labels


def write_ubyte_images(images, out_path):
    """
    Store images in ubyte format at the given path.
    Arguments:
        images (numpy.array): A uint8 array of images.
        out_path       (str): The output path.
    """
    if images.dtype != np.uint8:
        print("Failure: images aren't uint8!")
        sys.exit(1)

    magic_number = 0x0803

    # Information for the header
    with gzip.open(out_path, 'wb') as file:
        # Header is 4 big endian 4 byte integers
        file.write(struct.pack('>IIII', magic_number, *images.shape))

        # Write images
        file.write(images.tobytes())


def write_ubyte_labels(labels, out_path):
    """
    Store labels in ubyte format at the given path.
    Arguments:
        labels (numpy.array): Labels (0-9).
        out_path       (str): The output path.
    """
    if labels.dtype != np.uint8:
        print("Failure: labels aren't uint8!")
        sys.exit(1)

    magic_number = 0x0801

    # Information for the header
    with gzip.open(out_path, 'wb') as file:
        # Header is 2 big endian 4 byte integers
        file.write(struct.pack('>II', magic_number, len(labels)))

        # Write labels
        file.write(labels.tobytes())
