import argparse
import numpy
import pandas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "basename",
        help="Base name of the csv to parse.")
    parser.add_argument(
        "--chunksize",
        required=False,
        default=10000,
        help="The number of lines to read at a time.")

    args = parser.parse_args()

    packed_chunks = []
    target_chunks = []

    # Read in a chunk at a time
    for chunk in pandas.read_csv(f"{args.basename}.csv", header=0, chunksize=args.chunksize):
        # The cut version is: cut INT03.csv -d"," --fields=2-33,482
        # Pack the 32-bit address
        binary_features = chunk.iloc[:, 1:33].to_numpy(dtype=numpy.uint8)
        packed_chunk = numpy.packbits(binary_features, axis=1)
        packed_chunks.append(packed_chunk)

        # Extract the last column (branch taken) as boolean
        target_chunk = chunk.iloc[:, 481].to_numpy(dtype=bool)
        target_chunks.append(target_chunk)

    # Combine all processed chunks into final arrays
    packed_features = numpy.vstack(packed_chunks)
    target_array = numpy.concatenate(target_chunks)

    print("Final packed features shape:", packed_features.shape)
    print("Final target array shape:", target_array.shape)

    # First address is:
    print("First address is:")
    print(numpy.unpackbits(packed_features[0]))

    # Last address is:
    print("Last address is:")
    print(numpy.unpackbits(packed_features[-1]))

    numpy.savez_compressed(
        f"{args.basename}.npz", features=packed_features, targets=target_array
    )

    # Reload like this:
    """
    # Load the compressed archive
    data = numpy.load(f"{args.basename}.npz")

    # Access individual arrays using the keys defined during saving
    packed_features = data["features"]
    target_array = data["targets"]

    print(packed_features.shape, target_array.shape)
    """
