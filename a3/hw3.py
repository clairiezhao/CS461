import sys
import numpy as np

def y_to_sign(y):
    if y == 0:
        return -1
    else:
        return 1

def sign_to_y(sign):
    if sign < 0:
        return 0
    else:
        return 1

if __name__ == "__main__":
    TABLE_SIZE = 1024
    HISTORY_LEN = 24

    # 2D table of perceptrons (one row per truncated address)
    W = np.zeros((TABLE_SIZE, HISTORY_LEN))
    biases = np.zeros(TABLE_SIZE)

    # Global History Register storing the last N outcomes (-1 or 1)
    history = np.full(HISTORY_LEN, -1.0)

    pred_y = -1
    pred_sign = 0
    idx = 0
    lr = 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            break

        # If line is an address
        if pred_y == -1:
            # Convert bit string to an integer and truncate via modulo
            addr_int = int(line.replace(" ", ""), 2)
            idx = (addr_int // 4) % TABLE_SIZE

            # Predict using the perceptron at idx and the global history
            pred_sign = biases[idx] + np.dot(W[idx], history)
            pred_y = sign_to_y(pred_sign)
            print(pred_y)

        # If line is a label (0 or 1)
        else:
            y = int(line)
            sign = y_to_sign(y)

            # Update weights if wrong
            if pred_y != y or pred_y == 0:
                W[idx] += lr * sign * history
                biases[idx] += lr * sign

            # Shift the newest outcome into the front of the history vector
            history = np.roll(history, 1)
            history[0] = sign
            pred_y = -1