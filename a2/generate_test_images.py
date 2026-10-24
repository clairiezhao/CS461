import argparse
import mnist_util
import numpy as np
import random
from PIL import Image
import torch

import torchvision
import yaml


def fixOrientation(img):
    """EMNIST data matches the original MNIST data, so it is flipped and rotated. Fix that."""
    return np.rot90(np.fliplr(img))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "emnist_root",
        help="Root of a directory structure where the data lives: EMNIST/raw/*ubyte.")
    parser.add_argument(
        "corpus",
        type=str,
        help="File to read a text corpus from.")
    parser.add_argument(
        "--output",
        type=str,
        default="test_string.png",
        help="Generated image string.")
    parser.add_argument(
        "--random_seed",
        required=False,
        type=int,
        default=86,
        help="The random seed.")

    args = parser.parse_args()

    # Seed all of the random number generators for repeatability.
    # Keep in mind though, that some algorithms are nondeterministic, so this
    # doesn't guarantee fully repeatable results.
    np.random.default_rng(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    print("Loading data")
    images = mnist_util.load_mnist_ubyte(f"{args.emnist_root}/EMNIST/raw/emnist-byclass-test-images-idx3-ubyte.gz")
    image_labels = mnist_util.load_mnist_labels(f"{args.emnist_root}/EMNIST/raw/emnist-byclass-test-labels-idx1-ubyte.gz")
    unique_classes = np.unique(image_labels)
    # Make sure the first class is 0
    if min(unique_classes) != 0:
        image_labels = image_labels.copy() - min(unique_classes)

    label_mapping = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',]

    # Read the sentence and then grab random digits to write it out
    try:
        with open(args.corpus, 'r', encoding='utf-8') as corpus:
            text = corpus.read()
    except FileNotFoundError:
        print(f"Error: The file '{args.corpus}' was not found.")
        sys.exit(1)

    # Remove all non alphabetic characters except for spaces.
    # Leave in uppercase letters
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = ''.join(c for c in text if (c.isalpha() or c == ' '))

    letter_imgs = []
    for c in text:
        if c == ' ':
            # Save an empty space
            letter_imgs.append(np.zeros((28, 28)))
        else:
            class_idx = label_mapping.index(c)
            # Find all images of the matching class
            matching = np.where(image_labels == class_idx)[0]
            chosen_idx = random.choice(matching)
            letter_imgs.append(fixOrientation(images[chosen_idx]))
    final_string = np.hstack(letter_imgs)

    Image.fromarray(final_string.astype(np.uint8)).save(args.output)

