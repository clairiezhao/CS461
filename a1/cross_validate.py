import argparse
import math
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# For repeatability
np.random.seed(100)

parser = argparse.ArgumentParser(
    description="461 homework 1."
)

parser.add_argument(
    "load_csv",
    default=None,
    type=str,
    help="Load data from the provided csv."
)
parser.add_argument(
    "--plot_data",
    action='store_true',
    default=False,
    required=False,
    help="Plot the points and decision boundary."
)
parser.add_argument(
    "--lr",
    required=False,
    type=float,
    default=0.01,
    help="learning rate"
)
parser.add_argument(
    "--epochs",
    required=False,
    type=int,
    default=1000,
    help="training epochs"
)


args = parser.parse_args()

def getData(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    y = data[:,0]
    X = data[:,1:]
    return X, y

X, y = getData(args.load_csv)
print(f"Sizes are {X.shape} and {y.shape}")
# Shuffle the data
#reorder = np.arange(y.shape[0])
#np.random.shuffle(reorder)
#X = X[reorder]
#y = y[reorder]

#######################
# This is the solving code.

def cross_entropy_loss(y_hat, y):
    # See https://docs.pytorch.org/docs/2.13/generated/torch.nn.CrossEntropyLoss.html
    epsilon = 1e-15
    # Prevent division by 0
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y * np.log(y_hat), axis=y_hat.ndim-1))

def sigmoid(eta):
    """
    The sigmoid function.
    """
    return 1 / (1 + np.exp(-eta))

class LogisticRegressor(BaseEstimator, ClassifierMixin):
    """An estimator that works with sklearn DecisionBoundaryDisplay.from_estimator."""
    def __init__(self, degree=1, cross_features=False, learning_rate=0.01, epochs=20, batch_size=None, l2=0.0, l1=0.0, standardize=False):
        """
        Initialize a class for least squares regression to fit a polynomial of the desired degree.

        Args:
            degree      (int): Degree of the polynomial to fit. Applies to every column in X.
            learning_rate (float): Update rate
            epochs (int): Number of times to iterate through the data
            batch_size   (int): Number of values to use in a minibatch. Plain gradient descent if None.
            l2         (float): Strength of the l2 norm to apply
            l1         (float): Strength of the l1 norm to apply
            standardize (bool): Standardize the x values
        """
        self.degree = degree
        self.cross_features = cross_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.l1 = l1
        self.standardize = standardize
        # Assume two classes.
        # Sigmoid threshold
        self.prob_function = sigmoid
        self.threshold = 0.5
        self.classes_ = np.array([0, 1])
        self._estimator_type = "classifier"

    def expandFeatures(self, X):
        # Don't create columns for bias.
        if self.cross_features:
            # Add x*y, x*z, etc columns
            X = np.concatenate((X, np.stack([X[i]*X[j] for j in range(1, X.shape[0]) for i in range(0, j)])))
        return np.stack([X[j]**i for i in range(1, self.degree+1) for j in range(X.shape[0])])

    def fit(self, X, y):
        """
        Performs least squares regression to fit a polynomial of the desired degree
        to the provided data by constructing and solving the normal equations.

        Args:
            x (numpy.ndarray): The x values. Dimensions should be [feature, sample]
            y (numpy.ndarray): The y values
        """
        # X should be in [feature, sample] order
        if self.standardize:
            X = X.T
            x_mean = np.mean(X, axis=0)
            x_std = np.std(X, axis=0) + 10e-5
            self.std_stats = (x_mean, x_std)
            X = ((X - x_mean)/(x_std)).T
        else:
            self.std_stats = None

        # Create the requested degree input
        x = self.expandFeatures(X)

        nfeatures, samples = x.shape
        #print(f"training on {samples} samples with {nfeatures} features.")
        w = np.zeros(nfeatures)
        b = 0


        # Using RMSProp for gradient updates
        rms_beta = 0.9
        sq_avg = 0

        # Gradient descent or SGD
        batch_size = self.batch_size
        if batch_size is None:
            batch_size = samples
        else:
            # We should shuffle if we use minibatches
            # If the data was ushuffled, a batch could consist of a single class.
            random_order = np.arange(samples)
            np.random.shuffle(random_order)
            x = x[:,random_order]
            y = y[random_order]
        batches = math.ceil(samples/batch_size)
        for i in range(self.epochs):
            # For eval
            correct = 0
            avg_loss = 0
            for batch in range(batches):
                begin = batch * batch_size
                end = (batch+1) * batch_size
                minibatch = x.T[begin:end]
                miniy = y[begin:end]
                # Get the predicted probabilities
                y_hat = self.prob_function(minibatch @ w + b)

                # Calculate gradients w.r.t. w and b
                dw = (1 / len(minibatch)) * minibatch.T @ (y_hat - miniy)
                db = (1 / len(minibatch)) * np.sum(y_hat - miniy)

                sq_avg = rms_beta * sq_avg + (1 - rms_beta)*dw**2
                # Add in l2 penalty after rms
                # Update w and bias
                w -= self.learning_rate * (dw / (np.sqrt(sq_avg) + 10e-5) + self.l2*w + self.l1*np.sign(w))
                b -= self.learning_rate * db

                avg_loss += cross_entropy_loss(y_hat, miniy)
            avg_loss = avg_loss / batches

            #print(f"epoch {i} Cross Entropy Loss is {avg_loss}")
            # Get the predictions
            y_hat = (self.prob_function(minibatch @ w + b) > self.threshold).astype(int)
            matches = np.sum(y_hat == y)
            #print(f"Accuracy {matches/y.shape[0]}")
            self.weights_ = w
            self.bias_ = b
            # Stop early if there is no loss
            if avg_loss < 10e-6:
                break

        return self

    def decision_function(self, X):
        # sklearn gives X in samples, features order
        if self.std_stats is not None:
            X = (X - self.std_stats[0])/self.std_stats[1]
        X = X.T
        # Create the requested degree input
        X = self.expandFeatures(X)
        return self.prob_function(X.T @ self.weights_ + self.bias_)

    def predict(self, X):
        return (self.decision_function(X) >= self.threshold).astype(int)

#### Plotting and/or analysis

def fitAndPlot(X, y, degree, use_cross, l2, l1, epochs, lr):

    clf = LogisticRegressor(degree=degree, cross_features=True, l2=l2, l1=l1, epochs=epochs, learning_rate=lr)
    clf.fit(X.T, y)
    y_hat = clf.predict(X)
    matches = np.sum(y_hat == y)
    accuracy = matches/y.shape[0]
    print(f"Accuracy is {accuracy}")

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()

    # Plot the decision boundary
    step = 0.2
    xx, yy = np.meshgrid(np.arange(X[:, 0].min()-0.5, X[:, 0].max()+0.5, step), np.arange(X[:, 1].min()-0.5, X[:, 1].max()+0.5, step))
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
    else:
        Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]
    Z = Z.reshape(xx.shape)
    # Colormap with Z
    # Use one of the diverging maps (https://matplotlib.org/stable/users/explain/colors/colormaps.html)
    ax.contour(xx, yy, Z, cmap=mpl.colormaps['coolwarm'], alpha=0.8)

    title_str = f"degree {degree} fit, use cross={use_cross},\nl2={l2}, l1={l1}, accuracy={accuracy}"

    plt.title(title_str, fontsize=20)
    plt.savefig(f"degree_{degree}_fit_l2_{l2}_l1_{l1}_use_cross_{use_cross}.svg", dpi=2*96)
    #plt.show()
    plt.clf()

# TODO
# This is where you should insert your cross validation algorithm.
# split into N batches, for batch in batches use as holdout set
# def cross validation, returns hyperparameters
# y: n rows, 1 col
# X: n rows, 2 col

def cross_validate(X, y, epochs, lr):
    N_FOLDS = 5
    # splits the data preserving class weights
    splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=461).split(X, y))
    USE_CROSS_LIST = [True, False]
    L2_LIST = [0.0, 0.1, 1.0]
    L1_LIST = [0.0, 0.1, 1.0]
    DEGREE_LIST = [1, 2, 3]
    comparison_rows = []
    for use_cross in USE_CROSS_LIST:
        for l2 in L2_LIST:
            for l1 in L1_LIST:
                for deg in DEGREE_LIST:
                    settings = {"degree": deg, "cross_features": use_cross, "l1": l1, "l2": l2}
                    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
                        model = LogisticRegressor(**settings, epochs=epochs, learning_rate=lr)
                        model.fit(X[train_idx].T, y[train_idx])
                        val_p = model.decision_function(X[val_idx])
                        comparison_rows.append({**settings, "Fold": fold,
                            "Validation accuracy": np.mean((val_p >= .5) == y[val_idx]),
                            "Validation BCE": cross_entropy_loss(val_p, y[val_idx])})
    comparison_scores = pd.DataFrame(comparison_rows)
    comparison_summary = comparison_scores.groupby(["cross_features", "l2", "l1", "degree"], as_index=False).agg(
        mean_validation_accuracy=("Validation accuracy", "mean"),
        std_validation_accuracy=("Validation accuracy", "std"),
        mean_validation_bce=("Validation BCE", "mean"))

    # Selection uses unrounded validation means, never the reserved test results.
    # An exact tie keeps the first row (lower degree, then lower L2 in this table).
    best_row = comparison_summary.loc[comparison_summary.mean_validation_accuracy.idxmax()]
    selected_config = {"degree": int(best_row.degree), "cross_features": bool(best_row.cross_features),
                    "l1": float(best_row.l1), "l2": float(best_row.l2), "accuracy": comparison_summary.mean_validation_accuracy.max()}
    return selected_config


# Analyze and plot.
if args.plot_data:
    use_cross = False
    l2 = 0.2
    l1 = 0.0
    degree = 1
    plt.subplots(figsize=(8,6))
    fitAndPlot(X, y, degree, use_cross, l2, l1, args.epochs, args.lr)
else:
    '''clf = LogisticRegressor(degree=degree, cross_features=use_cross, l2=l2, l1=l1, epochs=args.epochs, learning_rate=args.lr)
    clf.fit(X.T, y)
    y_hat = clf.predict(X)
    matches = np.sum(y_hat == y)
    accuracy = matches/y.shape[0]
    print(f"Accuracy is {accuracy}")'''
    selected_config = cross_validate(X, y, args.epochs, args.lr)
    print(f"{selected_config["cross_features"]}, {selected_config["l2"]}, {selected_config["l1"]}, {selected_config["degree"]}")
    print(f"accuracy: {selected_config["accuracy"]}")