import argparse
import numpy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "basename",
        help="Base name of the csv to parse.")

    args = parser.parse_args()

    # Load the compressed archive
    data = numpy.load(f"{args.basename}.npz")

    # Access individual arrays using the keys defined during saving
    packed_features = data["features"]
    target_array = data["targets"]

    print("Packed features shape:", packed_features.shape)
    print("Target array shape:", target_array.shape)

    # First address is:
    print("First address is:")
    print(numpy.unpackbits(packed_features[0]))

    # Last address is:
    print("Last address is:")
    print(numpy.unpackbits(packed_features[-1]))

