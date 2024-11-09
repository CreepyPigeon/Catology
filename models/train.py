import os
import yaml
import numpy as np
from dotenv import load_dotenv
from architecture import NeuralNetwork, cross_entropy_loss
from data.processed.data_processing import get_loaders

load_dotenv()
results_dir = os.getenv('RESULTS_DIR')
errors_file = os.getenv('ERRORS_FILE')
train_data = 'balanced_train_data'

def normalize_data(X):
    """
    Normalize the input data to the range [0, 1].
    """
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

def import_hyperparameters():
    with open('hyperparameters.yaml', 'r') as f:
        loaded_hyperparameters = yaml.load(f, Loader=yaml.FullLoader)

    print("Loaded Hyperparameters:", loaded_hyperparameters)

    input_size = loaded_hyperparameters['input_size']
    hidden_size = loaded_hyperparameters['hidden_size']
    output_size = loaded_hyperparameters['output_size']
    learning_rate = loaded_hyperparameters['learning_rate']
    batch_size = loaded_hyperparameters['batch_size']

    return input_size, hidden_size, output_size, learning_rate, batch_size

def train_one_epoch(model, X_train, y_train, batch_size):
    model.train(X_train, y_train, epochs=1, batch_size=batch_size)

def train(epochs, weights_file='trained_weights.npz'):
    # Load the hyperparameters
    input_size, hidden_size, output_size, learning_rate, batch_size = import_hyperparameters()

    # Load the data
    X_train, y_train, X_val, y_val, race_desc_train, race_desc_val = get_loaders("balanced_train_data.xlsx")

    # Ensure data is in numpy array format
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = normalize_data(X_train)
    X_val = normalize_data(X_val)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # Initialize the model with the loaded hyperparameters
    model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save the weights after training
    model.save_weights(weights_file)

    # After training, evaluate the model on the validation set
    val_loss, val_accuracy = evaluate_model(model, X_val, y_val)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return model, X_val, y_val


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


if __name__ == '__main__':

    train(50)
