from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from data.processed.data_processing import get_loaders


if __name__ == '__main__':
    file_path = "balanced_train_data.xlsx"  # Replace with the actual path
    X_train, y_train, X_val, y_val, race_desc_train, race_desc_val = get_loaders(file_path)

    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees by default

    # Train the Random Forest model on the training set
    random_forest.fit(X_train, y_train)

    # Predictions on the validation set
    y_val_pred = random_forest.predict(X_val)

    # Accuracy
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=race_desc_val.unique()))
