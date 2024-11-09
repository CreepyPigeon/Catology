import os
import dotenv
import numpy as np

from data.processed.data_processing import add_synthetic_data, get_loaders
from models.architecture import NeuralNetwork

from sklearn.model_selection import KFold
from models.train import evaluate_model, train, import_hyperparameters

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
    input_size, hidden_size, output_size, learning_rate, batch_size = import_hyperparameters()

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


if __name__ == '__main__':
    file_path = train_data

    """
    X_train, y_train, X_val, y_val, race_desc_train, race_desc_val = get_loaders(train_data)

    # Display samples from the training set
    print("Sample of X_train:\n", X_train.head())  # First few rows of training features
    print("\nSample of y_train:\n", y_train.head())  # First few rows of training labels
    print("\nSample of race_desc_train:\n", race_desc_train.head())  # First few rows of race descriptions for training

    # Display samples from the validation set
    print("\nSample of X_val:\n", X_val.head())  # First few rows of validation features
    print("\nSample of y_val:\n", y_val.head())  # First few rows of validation labels
    print("\nSample of race_desc_val:\n", race_desc_val.head())  # First few rows of race descriptions for validation
    """

    # add_synthetic_data()

    # train(epochs=50, train_model=False)

    X_train, y_train, X_val, y_val, race_desc_train, race_desc_val = get_loaders('balanced_train_data.xlsx')
    cross_validate(model_class=NeuralNetwork, X=X_val, y=y_val, num_folds=5)
