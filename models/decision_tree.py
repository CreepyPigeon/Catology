import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from data.processed.data_processing import split_data, load_data

if __name__ == '__main__':
    file_path = "../data/processed/balanced_train_data.xlsx"
    X, y, race_desc = load_data(file_path)

    X = np.array(X)
    y = np.array(y)

    X_train, y_train, X_val, y_val = split_data(X, y, method='train_test')
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees by default

    # Train the Random Forest model on the training set
    random_forest.fit(X_train, y_train)

    # Predictions on the validation set
    y_val_pred = random_forest.predict(X_val)

    # Accuracy
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    # print("Classification Report:")
    # print(classification_report(y_val, y_val_pred, target_names=np.unique(y)))
