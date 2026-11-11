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
    parser.add_argument(
        "--ignore",
        type=int,
        required=False,
        default=100,
        help="Number of samples to ignore.")

    args = parser.parse_args()

    # Load the compressed archive
    data = numpy.load(f"{args.basename}.npz")

    # Access individual arrays using the keys defined during saving
    packed_features = data["features"]
    target_array = data["targets"]

    correct = 0
    total = 0
    ignore_left = args.ignore
    for i in range(args.begin, args.end):
        y = int(target_array[i])
        y_hat_str = sys.stdin.readline().strip()
        if not(y_hat_str.isnumeric()) or 1 != len(y_hat_str):
            print(f"Error with program output. Received: '{y_hat_str}'")
            break
        y_hat = int(y_hat_str)
        if y_hat not in [0, 1]:
            print("Got a prediction that is not 0 or 1.")
            break
        if ignore_left == 0:
            if y_hat == y:
                correct += 1
            total += 1
        else:
            ignore_left -= 1
    print(f"Accuracy: {correct/total}")

