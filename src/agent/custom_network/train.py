import os
import yaml
import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from src.agent.custom_network.architecture import NeuralNetwork, cross_entropy_loss
from data.processed.data_processing import load_data, split_data
from src.utils.plot_utils import plot_fold_accuracy, plot_all_folds

load_dotenv()
results_dir = os.getenv('RESULTS_DIR')
errors_file = os.getenv('ERRORS_FILE')
train_data = os.getenv('BALANCED_DATASET')
weights_path = os.getenv('WEIGHTS')


def normalize_data(X):
    """
    Normalize the input data to the range [0, 1].
    """
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))


def import_hyperparameters(file='hyperparameters.yaml'):
    with open(file, 'r') as f:
        loaded_hyperparameters = yaml.load(f, Loader=yaml.FullLoader)

    print("Loaded Hyperparameters:", loaded_hyperparameters)

    epochs = loaded_hyperparameters['epochs']
    input_size = loaded_hyperparameters['input_size']
    hidden_size = loaded_hyperparameters['hidden_size']
    output_size = loaded_hyperparameters['output_size']
    learning_rate = loaded_hyperparameters['learning_rate']
    batch_size = loaded_hyperparameters['batch_size']

    return epochs, input_size, hidden_size, output_size, learning_rate, batch_size


def visualize_mismatches(model, X, y, weights_file):
    """
    Visualize mismatched instances using t-SNE.

    Arguments:
    - model: Trained model
    - X: Validation features
    - y: Validation labels
    - weights_file: Path to model weights for predictions

    Returns:
    - None (displays a scatter plot)
    """
    # Load model weights (optional if already loaded)
    model.load_weights(weights_file)

    # Predict validation set
    y_pred = model.predict(X)

    # Identify mismatched instances
    mismatches = y != y_pred
    X_mismatched = X[mismatches]
    y_true_mismatched = y[mismatches]
    y_pred_mismatched = y_pred[mismatches]

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_mismatched)

    # Scatter plot
    plt.figure(figsize=(10, 8))
    for true_label in np.unique(y_true_mismatched):
        indices = (y_true_mismatched == true_label)
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=f"True: {true_label}", alpha=0.6)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred_mismatched, cmap='coolwarm', marker='x', label='Predicted')
    plt.title("t-SNE Visualization of Mismatched Instances")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.show()

def train(weights_file=weights_path, split_method='train_test', num_folds=5,
          save_path='training_loss_80/20.png'):
    """
    Train the model with specified data splitting method (train-test or cross-validation).
    Arguments:
    - file_path: Path to the dataset file
    - weights_file: Path to save the trained model weights
    - split_method: Splitting method ('train_test' or 'cross_validation')
    - num_folds: Number of folds (used only for cross-validation)

    Returns:
    - Trained model and validation data (for train-test split)
    """
    epochs, input_size, hidden_size, output_size, learning_rate, batch_size = import_hyperparameters()

    X, y, race_desc = load_data(train_data)
    X = np.array(X)
    y = np.array(y)

    if split_method == 'train_test':
        X_train, y_train, X_val, y_val = split_data(X, y, method='train_test')
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        # X_train = normalize_data(X_train)
        # X_val = normalize_data(X_val)

        model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
        model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, save_path=save_path)
        model.save_weights(weights_file)

        val_loss, val_accuracy = evaluate_model(model, X_val, y_val, weights_file=weights_file)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        visualize_mismatches(model, X_val, y_val, weights_file)
        return model, X_val, y_val

    # Cross-validation
    elif split_method == 'cross_validation':
        return train_with_cross_validation(num_folds=num_folds, weights_file=weights_file)

    else:
        raise ValueError("Invalid split_method. Use 'train_test' or 'cross_validation'.")


def evaluate_model(model, X_val, y_val, weights_file=weights_path):
    """
    Evaluate the model on validation data by loading weights from the saved file.
    Arguments:
    - model: The model class instance
    - X_val: Validation input data
    - y_val: Validation labels
    - weights_file: Path to the file where the model's weights are saved
    - lambda_reg: Regularization strength

    Returns:
    - val_loss: The loss value for the validation set
    - val_accuracy: The accuracy for the validation set
    """
    # load the trained weights from the file
    # model.load_weights(weights_file)

    y_pred = model.forward(X_val)

    val_loss = cross_entropy_loss(y_val, y_pred)

    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y_val)
    accuracy *= 100

    return val_loss, accuracy

def train_with_cross_validation(num_folds=5, weights_file='trained_weights.npz'):
    """
    Train the model using cross-validation and save loss plots for each fold.

    Arguments:
    - num_folds: Number of folds (used for cross-validation)
    - weights_file: Path to save the trained model weights
    - save_path: Path where to save the plot (can be modified for different fold-specific plots)

    Returns:
    - None
    """
    X, y, race_desc = load_data(train_data)
    X = np.array(X)
    y = np.array(y)

    epochs, input_size, hidden_size, output_size, learning_rate, batch_size = import_hyperparameters()
    kf = KFold(n_splits=num_folds)

    fold = 0
    fold_metrics = []
    fold_accuracies = []

    # folder for saving plots if it doesn't exist
    plot_folder = 'cv_loss_plots'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    for train_index, val_index in kf.split(X):  # kf.split(X) to get train/val indices
        fold += 1
        print(f"Training fold {fold}/{num_folds}...")

        # Split data into training and validation for this fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # a new model instance for each fold
        model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                              learning_rate=learning_rate)

        fold_training_accuracies = []

        for epoch in range(epochs):
            model.train(X_train, y_train, 1, batch_size=batch_size, save_path=None)
            val_loss, val_accuracy = evaluate_model(model, X_val, y_val)
            fold_training_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}")
        fold_accuracies.append(fold_training_accuracies)

        # evaluate the model on the final validation set
        val_loss, val_accuracy = evaluate_model(model, X_val, y_val)
        print(f"Fold {fold} - Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_accuracy:.4f}")

        fold_metrics.append((val_loss, val_accuracy))
        model.save_weights(weights_file)

        fold_save_path = os.path.join(plot_folder, f'fold_{fold}_accuracy.png')
        plot_fold_accuracy(fold_training_accuracies, fold_save_path)

    # average metrics across all folds
    avg_loss = np.mean([m[0] for m in fold_metrics])
    avg_accuracy = np.mean([m[1] for m in fold_metrics])
    print(f"Cross-Validation Completed. Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

    plot_all_folds(fold_accuracies, plot_folder)


if __name__ == '__main__':
    train(weights_file='../trained_weights.npz', split_method='train_test',
          save_path='../cv_loss_plots/training_loss_80_20.png')

    # train(weights_file=weights_path, split_method='cross_validation')
