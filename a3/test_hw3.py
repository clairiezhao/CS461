import argparse
import numpy
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "basename",
        help="Base name of the csv to parse.")
    parser.add_argument(
        "--begin",
        type=int,
        required=True,
        help="Line to begin.")
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="Line to end.")

    args = parser.parse_args()

    # Load the compressed archive
    data = numpy.load(f"{args.basename}.npz")

    # Access individual arrays using the keys defined during saving
    packed_features = data["features"]
    target_array = data["targets"]

    for i in range(args.begin, args.end):
        print(' '.join(map(str, numpy.unpackbits(packed_features[i]))))
        print(int(target_array[i]))

