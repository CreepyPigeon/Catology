import os
import dotenv
import numpy as np

from data.processed.data_processing import load_data
from models.architecture import NeuralNetwork

from sklearn.model_selection import KFold
from models.train import evaluate_model, import_hyperparameters

dotenv.load_dotenv()
train_data = os.getenv('TRAIN_DATA')


def cross_validate(model_class, X, y, num_folds=5, weights_file='trained_weights.npz'):
    """
    Perform cross-validation for the given model.
    Arguments:
    - model_class: The model class to be used (e.g., NeuralNetwork)
    - X: Features (Pandas DataFrame)
    - y: Labels (Pandas Series)
    - num_folds: Number of folds for cross-validation
    - param_file: Path to the file containing the hyperparameters
    - weights_file: Path to the saved weights file to load during evaluation

    Returns:
    - None (This function prints out validation loss and accuracy for each fold)
    """
    # Load the hyperparameters from the file (this will replace the static ones)
    _, input_size, hidden_size, output_size, learning_rate, batch_size = import_hyperparameters()

    # Initialize the model with the loaded hyperparameters
    model = model_class(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    # Split data into k-folds for cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold = 0
    for train_index, val_index in kf.split(X):
        fold += 1
        print(f"Evaluating fold {fold}...")

        # Split the data into train and validation sets for this fold
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]  # Use iloc to index by rows
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]  # Same for labels

        # Here, you could optionally train the model using the training data (not shown here)
        # But, for evaluation, we are assuming the weights have already been trained and saved to 'trained_weights.npz'

        # Evaluate the model using the saved weights
        val_loss, val_accuracy = evaluate_model(model, X_val, y_val, weights_file=weights_file)
        print(f"Fold {fold} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


def check_prediction(model, X_input, y_input, weights_file):
    model.load_weights(weights_file)

    X_input = X_input.reshape(1, -1)  # Reshape to a 2D array (batch size of 1)

    raw_output = model.forward(X_input)

    # Print the raw output (before applying np.argmax)
    print(f"Raw output (logits) for the input: {raw_output}")

    # Get the model's prediction (class with highest probability)
    prediction = np.argmax(raw_output, axis=1)

    # Print the prediction and compare with the true value
    # print(f"Predicted class: {prediction[0]}, True class: {y_input}")

    # Check if the prediction matches the true class
    print(f"Prediciton {prediction[0]}")
    if prediction[0] == y_input:
        print(f"Prediction is correct!")
    else:
        print(f"Prediction is incorrect.")


if __name__ == '__main__':
    file_path = train_data

    epochs, input_size, hidden_size, output_size, learning_rate, batch_size = import_hyperparameters()
    nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                       learning_rate=learning_rate)

    weights = 'trained_weights.npz'
    X_input = np.array(
        [1, 1.5, 2, 2, 1, 0, 2, 3, 4, 1, 4, 4, 4, 5, 4, 3, 1, 1, 1, 1, 4, 1, 2, 0, 0])
    y_out = 7
    check_prediction(nn, X_input, y_out, weights)
