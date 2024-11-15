import os
import yaml
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import KFold

from architecture import NeuralNetwork, cross_entropy_loss
from data.processed.data_processing import load_data, split_data
from models.plot_utils import plot_fold_accuracy, plot_convergence_accuracy

load_dotenv()
results_dir = os.getenv('RESULTS_DIR')
errors_file = os.getenv('ERRORS_FILE')
train_data = 'balanced_train_data.xlsx'


def normalize_data(X):
    """
    Normalize the input data to the range [0, 1].
    """
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))


def import_hyperparameters():
    with open('hyperparameters.yaml', 'r') as f:
        loaded_hyperparameters = yaml.load(f, Loader=yaml.FullLoader)

    print("Loaded Hyperparameters:", loaded_hyperparameters)

    epochs = loaded_hyperparameters['epochs']
    input_size = loaded_hyperparameters['input_size']
    hidden_size = loaded_hyperparameters['hidden_size']
    output_size = loaded_hyperparameters['output_size']
    learning_rate = loaded_hyperparameters['learning_rate']
    batch_size = loaded_hyperparameters['batch_size']

    return epochs, input_size, hidden_size, output_size, learning_rate, batch_size


def train(weights_file='trained_weights.npz', split_method='train_test', num_folds=10,
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
    # Load hyperparameters
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
        model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, save_path=save_path)  # Save plot here
        model.save_weights(weights_file)

        val_loss, val_accuracy = evaluate_model(model, X_val, y_val, weights_file=weights_file)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        return model, X_val, y_val

    # Cross-validation
    elif split_method == 'cross_validation':
        return train_with_cross_validation(num_folds=num_folds, weights_file=weights_file, save_path=save_path)

    else:
        raise ValueError("Invalid split_method. Use 'train_test' or 'cross_validation'.")


def evaluate_model(model, X_val, y_val, weights_file='trained_weights.npz', lambda_reg=0.01):
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
    # Load the trained weights from the file
    model.load_weights(weights_file)

    # Forward pass
    y_pred = model.forward(X_val)

    # Compute loss with regularization
    val_loss = cross_entropy_loss(y_val, y_pred)  # Cross-entropy loss

    # Calculate accuracy
    predictions = np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y_val)

    return val_loss, accuracy


def train_with_cross_validation(num_folds=5, weights_file='trained_weights.npz',
                                save_path='cv_loss_plots/cross_val_loss_plot.png'):
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
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold = 0
    fold_metrics = []
    fold_accuracies = []  # Store accuracies for visualization

    plot_folder = 'cv_loss_plots'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    for train_index, val_index in kf:
        fold += 1
        print(f"Training fold {fold}/{num_folds}...")

        # Split data into training and validation for this fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Initialize a new model instance for each fold
        model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                              learning_rate=learning_rate)

        fold_training_accuracies = []

        for epoch in range(epochs):
            # Train for one epoch and get the loss, pass save_path for saving the plot (no plotting here)
            epoch_losses = model.train(X_train, y_train, 1, batch_size=batch_size, save_path=None)  # Don't save a plot here

            # Evaluate on validation set
            val_loss, val_accuracy = evaluate_model(model, X_val, y_val)

            # Store the accuracy after each epoch
            fold_training_accuracies.append(val_accuracy)

            # Print accuracy after each epoch (train and val accuracy)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}")

        # After all epochs, store the fold accuracy for plotting
        fold_accuracies.append(fold_training_accuracies)

        # Evaluate the model on the validation set (final evaluation after training)
        val_loss, val_accuracy = evaluate_model(model, X_val, y_val)
        print(f"Fold {fold} - Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_accuracy:.4f}")

        fold_metrics.append((val_loss, val_accuracy))

        # Save weights after training
        model.save_weights(weights_file)

        # Save plot for the fold only after finishing all epochs for that fold
        fold_save_path = os.path.join(plot_folder, f'fold_{fold}_accuracy_plot.png')
        plot_fold_accuracy(fold_training_accuracies, fold_save_path)

    # Compute average metrics across all folds
    avg_loss = np.mean([m[0] for m in fold_metrics])
    avg_accuracy = np.mean([m[1] for m in fold_metrics])
    print(f"Cross-Validation Completed. Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

    # Plot average accuracy for all folds after all are finished
    plot_convergence_accuracy(fold_accuracies, save_path)


if __name__ == '__main__':
    # train(weights_file='trained_weights.npz', split_method='train_test', save_path='training_loss_80_20.png')

    train(weights_file='trained_weights.npz', split_method='cross_validation', save_path='cross_val_loss.png')
