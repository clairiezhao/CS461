import argparse
import numpy as np

# For repeatability
np.random.seed(100)

parser = argparse.ArgumentParser(
    description="461 homework 1 data generator."
)

parser.add_argument(
    "--num_samples",
    type=int,
    default=20,
    required=False,
    help="Number of data samples to generate (for each class)."
)
parser.add_argument(
    "--dataset_type",
    choices=['a', 'b', 'c'],
    default='a',
    required=False,
    help="Which dataset to use. The first is sun and moon. The second a circle inside of a ring, with noise that causes overlap. The last is a noiseless 4 squares."
)

args = parser.parse_args()
num_samples = args.num_samples

def genDataA(num_samples):
    """Returns X and Y data for the sun and moon dataset."""
    # Something similar to two moons
    # One class is a central cluster (the sun), the other is an arc (the Moon)
    # Moon is class 0. Has uniform angle and normal x noise with laplace y noise
    # Sun is class 1. Has laplace x and gaussian y.
    # Radius of the arc class
    R = 10
    moon_noise = 1
    sun_noise = 2
    random_angles = np.random.uniform(np.pi/10, 7*np.pi/10, num_samples)
    moon = np.stack((R*np.cos(random_angles), R*np.sin(random_angles)), axis=1) + np.stack((np.random.normal(0, moon_noise, num_samples), np.random.laplace(-1, moon_noise, num_samples)), axis=1)
    sun = np.stack((np.random.laplace(0, sun_noise, num_samples), np.random.normal(2, sun_noise, num_samples)), axis=1)

    X = np.concatenate((moon, sun))
    y = np.array([0]*num_samples + [1]*num_samples)
    return X, y

def genDataB(num_samples):
    """Returns X and Y data for a circle inside of a ring."""
    R = 10
    noise_std = 1
    random_angles = np.random.uniform(0, 2*np.pi, num_samples)
    ring = np.stack((R*np.cos(random_angles), R*np.sin(random_angles)), axis=1) + np.stack((np.random.normal(0, noise_std, num_samples), np.random.normal(0, noise_std, num_samples)), axis=1)

    random_angles = np.random.uniform(0, 2*np.pi, num_samples)
    random_radii = np.random.uniform(0, R, num_samples)
    circle = np.stack((random_radii*np.cos(random_angles), random_radii*np.sin(random_angles)), axis=1) + np.stack((np.random.normal(0, noise_std, num_samples), np.random.normal(0, noise_std, num_samples)), axis=1)

    X = np.concatenate((ring, circle))
    y = np.array([0]*num_samples + [1]*num_samples)
    return X, y

def genDataC(num_samples):
    """Returns X and Y data for XOR data."""
    # class a has signs: sign(y) = -sign(x)
    datarange = 1
    quadrants = np.random.choice([-1, 1], size=num_samples)
    class_a = np.stack((quadrants * np.random.uniform(0, datarange, num_samples), -1*quadrants * np.random.uniform(0, datarange, num_samples)), axis=1)
    # class b has signs: sign(y) = sign(x)
    quadrants = np.random.choice([-1, 1], size=num_samples)
    class_b = np.stack((quadrants * np.random.uniform(0, datarange, num_samples), quadrants * np.random.uniform(0, datarange, num_samples)), axis=1)

    X = np.concatenate((class_a, class_b))
    y = np.array([0]*num_samples + [1]*num_samples)
    return X, y

def getData(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    y = data[0]
    X = data[1:]
    return X, y

if args.dataset_type == 'a':
    X, y = genDataA(args.num_samples)
elif args.dataset_type == 'b':
    X, y = genDataB(args.num_samples)
elif args.dataset_type == 'c':
    X, y = genDataC(args.num_samples)
# Shuffle the data
reorder = np.arange(y.shape[0])
np.random.shuffle(reorder)
X = X[reorder]
y = y[reorder]

data = np.concatenate((y.reshape(-1, 1), X), axis=1)
# Print the header
print("class, x, y")
for row in data:
    print(', '.join([str(val) for val in row]))
