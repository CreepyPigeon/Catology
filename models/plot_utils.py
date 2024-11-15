import os

import numpy as np
from matplotlib import pyplot as plt


def plot_fold_accuracy(accuracies, save_path):
    """
    Plot and save the accuracy for a single fold.

    Arguments:
    - accuracies: List of accuracy values to plot
    - save_path: Where to save the plot
    """
    plt.plot(range(1, len(accuracies) + 1), accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_convergence_accuracy(accuracies, save_path):
    """
    Plot the convergence of accuracy across all folds.

    Arguments:
    - accuracies: List of accuracies for each fold
    - save_path: Where to save the plot
    """
    plt.figure(figsize=(10, 6))
    for i, fold_accuracy in enumerate(accuracies):
        plt.plot(range(1, len(fold_accuracy) + 1), fold_accuracy, label=f"Fold {i + 1}")

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Cross-validation Accuracy per Epoch')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_fold_loss(training_loss, validation_loss, save_path):
    """
    Plot the training and validation loss for a single fold and save it to a file.

    Arguments:
    - training_loss: List of training loss values
    - validation_loss: List of validation loss values
    - save_path: Path where to save the plot

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label='Training Loss', color='blue')
    plt.plot(validation_loss, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Fold)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_convergence(training_losses, validation_losses, epochs, save_path=None):
    """
    Plot the training and validation loss curves.

    Arguments:
    - training_losses: List of lists containing training loss per fold
    - validation_losses: List of lists containing validation loss per fold
    - epochs: Number of epochs
    - save_path: Path to save the plot image (optional)

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))

    # Plot training loss for each fold
    for i, fold_loss in enumerate(training_losses):
        plt.plot(range(epochs), fold_loss, label=f"Train Fold {i + 1}")

    # Plot validation loss for each fold
    for i, fold_loss in enumerate(validation_losses):
        plt.plot(range(epochs), fold_loss, label=f"Val Fold {i + 1}", linestyle='--')

    plt.title("Training and Validation Loss per Epoch (Cross-validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved as {save_path}")
    else:
        plt.show()  # If no save path, just display the plot

    plt.close()

def plot_and_save_loss(epoch_losses, save_path='loss_curve.png'):
    if not os.path.exists(save_path):
        # Plot the loss curve
        plt.plot(epoch_losses)
        plt.title("Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        # Save the plot to the specified file
        plt.savefig(save_path)
        print(f"Loss curve saved as {save_path}")
        plt.close()  # Close the plot to free up memory
    else:
        print(f"Plot already exists: {save_path}")

def plot_misclassified_points(X, y, model):
    y_pred = model.predict(X)
    misclassified_idx = np.where(y_pred != y)[0]

    # Assume features can be reduced to 2D using PCA or are already 2D
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    plt.figure(figsize=(10, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="viridis", alpha=0.6, label="Correctly Classified")
    plt.scatter(X_reduced[misclassified_idx, 0], X_reduced[misclassified_idx, 1], c="red", label="Misclassified",
                edgecolor="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Misclassified Points Visualization")
    plt.legend()
    plt.grid()
    plt.show()
