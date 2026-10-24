import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from sklearn.metrics import ConfusionMatrixDisplay

import torchvision
import yaml

import example_models


class fixOrientation:
    """EMNIST data matches the original MNIST data, so it is flipped and rotated. Fix that."""

    def __call__(self, img):
        return torchvision.transforms.functional.hflip(torchvision.transforms.functional.rotate(img, -90))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emnist_root",
        required=True,
        help="Root of a directory structure where the data lives: EMNIST/raw/*ubyte.")
    parser.add_argument(
        "--load",
        required=False,
        type=str,
        default=None,
        help="Skipping training and load an existing model to print out statistics.")
    parser.add_argument(
        "--epochs",
        required=False,
        type=int,
        default=10,
        help="Number of epochs to train.")
    parser.add_argument(
        "--train_samples",
        required=False,
        type=int,
        default=None,
        help="Number of samples to use for training, or all if None.")
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=16,
        help="The batch size.")
    parser.add_argument(
        "--random_seed",
        required=False,
        type=int,
        default=112,
        help="The random seed.")
    parser.add_argument(
        "--device",
        required=False,
        type=str,
        default=None,
        help="Override the automatically determined device (cuda or cpu).")
    parser.add_argument(
        "--weight_decay",
        required=False,
        type=float,
        default=0.0,
        help="Weight decay.")
    parser.add_argument(
        "--version",
        required=False,
        choices=["convnet", "wider", "deeper", "next"],
        default="convnet",
        help="Version of the convnet.")

    args = parser.parse_args()

    # Seed all of the random number generators for repeatability.
    # Keep in mind though, that some algorithms are nondeterministic, so this
    # doesn't guarantee fully repeatable results.
    np.random.default_rng(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading data")
    # Use the ToTensor transform, which converts [0-255] PIL images into [0-1] tensors.
    transforms = torchvision.transforms.Compose([
        fixOrientation(),
        torchvision.transforms.ToTensor()])

    # Use the letters split
    train_emnist = torchvision.datasets.EMNIST(root=args.emnist_root, split="letters", train=True, download=False, transform=transforms)
    test_emnist = torchvision.datasets.EMNIST(root=args.emnist_root, split="letters", train=False, download=False, transform=transforms)
    # Extract the labels
    Y_train_emnist = [sample[1]-1 for sample in train_emnist]
    Y_test_emnist = [sample[1]-1 for sample in test_emnist]
    unique_classes = np.unique(Y_test_emnist)
    total_classes = len(unique_classes)
    print(f"Training with {total_classes} training classes.")

    ## If you want to save some emnist
    #for i in range (20):
    #    torchvision.transforms.functional.to_pil_image(test_emnist[i][0]).save(f"example_test_emnist_{i}.png")

    # Create the model
    if args.version == 'convnet':
        model = example_models.ConvNet(classes=total_classes)
    elif args.version == 'wider':
        model = example_models.WiderConvNet(classes=total_classes)
    elif args.version == 'deeper':
        model = example_models.DeeperConvNet(classes=total_classes)
    elif args.version == 'next':
        model = example_models.ConvNeXtEMNIST(classes=total_classes)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params}")

    test_batch_size = 256

    # Are we doing training, or just reloading?
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    training_loss = []
    testing_loss = []
    training_acc = []
    testing_acc = []

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_emnist,
        batch_size=args.batch_size,
        # Note: I do not like shuffling each time.
        # In a professional setting we would shuffle this once on disk.
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_emnist,
        batch_size=test_batch_size,
        # Note: I do not like shuffling each time.
        # In a professional setting we would shuffle this once on disk.
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    if args.load is None:

        print("Training starting.")

        for epoch in range(args.epochs):

            total_loss = 0.0
            matches = 0
            mismatches = 0
            model.train()
            for X_batch, Y_batch in train_loader:
                # Classes begin counting from 1
                Y_batch = Y_batch - 1
                X_batch = X_batch.to(device, non_blocking=True)
                Y_batch = Y_batch.to(device, non_blocking=True)
                # Zero gradients before gradient calculation
                optimizer.zero_grad()

                y_hat = model(X_batch)
                loss = criterion(y_hat, Y_batch)
                total_loss += loss.item() * X_batch.size(0)

                # Gradient calculation
                loss.backward()
                # Update weights
                optimizer.step()
                # Accuracy check
                classes = torch.argmax(y_hat, dim=1)
                matches += torch.sum(classes == Y_batch)
                mismatches += torch.sum(classes != Y_batch)

            epoch_loss = total_loss / len(Y_train_emnist)
            print(f"Epoch {epoch} training loss {epoch_loss}")
            training_loss.append(epoch_loss)
            train_accuracy = matches / len(Y_train_emnist)

            # Evaluation
            # Don't calculate gradients during these steps
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                ## Accuracy values
                matches = 0
                mismatches = 0
                for X_batch, Y_batch in test_loader:
                    Y_batch = Y_batch - 1
                    X_batch = X_batch.to(device, non_blocking=True)
                    Y_batch = Y_batch.to(device, non_blocking=True)

                    y_hat = model(X_batch)
                    loss = criterion(y_hat, Y_batch)
                    total_loss += loss.item() * X_batch.size(0)
                    classes = torch.argmax(y_hat, dim=1)
                    matches += torch.sum(classes == Y_batch)
                    mismatches += torch.sum(classes != Y_batch)
                epoch_loss = total_loss / len(Y_test_emnist)
                print(f"Epoch {epoch} testing loss {epoch_loss}")
                testing_loss.append(epoch_loss)
                test_accuracy = matches / len(Y_test_emnist)
                print(f"Epoch {epoch} accuracies are {train_accuracy} {test_accuracy}")
                training_acc.append(train_accuracy.cpu().item())
                testing_acc.append(test_accuracy.cpu().item())
        torch.save(model.state_dict(), f"{args.version}_model.pyt")

        colors=["#2E2585", "#337538", "#5DA899", "#94CBEC"]

        size = plt.rcParams['lines.markersize'] ** 2

        epochs = list(range(args.epochs))

        plt.title("EMNIST Loss")
        plt.plot(epochs, training_loss, 'ro-', label="Training", color=colors[1], linewidth=3, linestyle="solid", alpha=0.95)
        plt.plot(epochs, testing_loss, 'ro-', label="Testing", color=colors[2], linewidth=3, linestyle="solid", alpha=0.95)

        plt.xlabel(r'Epoch', fontsize=16)
        plt.ylabel(r'Cross Entropy Loss', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.legend()

        plt.savefig(f"{args.version}_emnist_loss.svg", dpi=2*96)

        # Clear the plot and graph the accuracy curve
        plt.cla()

        plt.title("EMNIST Accuracy")
        plt.plot(epochs, training_acc, 'ro-', label="Training", color=colors[1], linewidth=3, linestyle="solid", alpha=0.95)
        plt.plot(epochs, testing_acc, 'ro-', label="Testing", color=colors[2], linewidth=3, linestyle="solid", alpha=0.95)

        plt.xlabel(r'Epoch', fontsize=16)
        plt.ylabel(r'Accuracy', fontsize=16)
        plt.ylim(None, 1.0)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.legend()

        plt.savefig(f"{args.version}_emnist_accuracy.svg", dpi=2*96)
    else:
        # Loading an existing model
        model.load_state_dict(torch.load(f"{args.version}_model.pyt", weights_only=True, map_location=device))
        model.to(device)

    ## Calculate and print the confusion matrix
    confusion = np.zeros((total_classes, total_classes))
    vocab, class_counts = np.unique(Y_test_emnist, return_counts=True)
    # Don't calculate gradients during these steps
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        ## Accuracy values
        matches = 0
        mismatches = 0
        for X_batch, Y_batch in test_loader:
            Y_batch = Y_batch - 1
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            y_hat = model(X_batch)
            loss = criterion(y_hat, Y_batch)
            total_loss += loss.item() * X_batch.size(0)
            classes = torch.argmax(y_hat, dim=1)
            for actual, predicted in zip(Y_batch, classes):
                confusion[actual,predicted] += 1
        for v,count in zip(vocab, class_counts):
            confusion[v] = confusion[v]/class_counts[v]

    # Clear the plot and graph the accuracy curve
    plt.cla()
    plt.title("EMNIST Confusion")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels = 'abcdefghijklmnopqrstuvwxyz')
    disp.plot(include_values=False)
    plt.savefig(f"{args.version}_emnist_confusion.svg", dpi=2*96)

    print(f"Vocab: {vocab}")
    yaml_config = {
        'model_version': args.version,
        'model': f"{args.version}_model.pyt",
        'confusion': confusion.tolist()}
    with open(f"{args.version}_config.yaml", 'w') as outfile:
        yaml.dump(yaml_config, outfile)

