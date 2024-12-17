import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

from data.processed.utils import print_fixed_length_message, save_to_excel, map_age, map_area, map_cats, map_sex, \
    map_place_of_living, write_errors

load_dotenv()
path = os.getenv('FILE_PATH')
results_dir = os.getenv('RESULTS_DIR')
errors_file = os.getenv('ERRORS_FILE')
new_data = os.getenv('NEW_DATA')
train_data = os.getenv('TRAIN_DATA')

data = pd.read_excel(path)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# 1st step
def remove_columns(original_dataset):
    df = pd.read_excel(original_dataset)
    df = df.drop(df.columns[[0, 1]], axis=1)
    df.to_excel(new_data, index=False)

    # once the removal is done, return a new file in xlsx format
    # this newly returned file will be used from now on
    df_modified = pd.read_excel(new_data)

    return df_modified


# 2nd step
def analyze_missing_values(df):
    missing_values = df.isnull().sum()
    unknown_values = (df == 'Unknown').sum()
    combined_report = missing_values + unknown_values
    write_errors("Missing or 'Unknown' Values:\n" + str(combined_report))


# 3rd step
def check_repeated_instances(df):
    repeated_instances = df[df.duplicated()]
    write_errors("Repeated Instances:\n" + str(repeated_instances))


# 4th
def check_and_drop_repeated_instances(df):
    repeated_instances = df[df.duplicated()]
    write_errors("Repeated Instances:\n" + str(repeated_instances))
    df = df.drop_duplicates()
    save_to_excel(df)
    return df


# 5th
def transform_abundance_column(df):
    df['Abundance of natural areas'] = df['Abundance of natural areas'].replace('Unknown', np.nan)
    df['Abundance of natural areas'] = pd.to_numeric(df['Abundance of natural areas'], errors='coerce')
    median_value = int(df['Abundance of natural areas'].median())
    df['Abundance of natural areas'] = df['Abundance of natural areas'].fillna(median_value)


# 6th
def check_and_drop_unknown_instances(df):
    unknown_instances = df[df.isin(['Unknown']).any(axis=1)]
    print(f"Rows with 'Unknown' values:\n{unknown_instances}")
    df = df.drop(unknown_instances.index)
    save_to_excel(df)
    return df


# 7th
def transform_place_of_living(df):
    df['Place of living'] = df['Place of living'].apply(map_place_of_living)
    save_to_excel(df)


# 8th
def transform_sex_column(df):
    df['Sex'] = df['Sex'].apply(map_sex)
    save_to_excel(df)


# 9th
def transform_area_column(df):
    df['Urban/Rural area'] = df['Urban/Rural area'].apply(map_area)
    save_to_excel(df)


# 10th
def transform_number_of_cats_column(df):
    df['Number of cats in the household'] = df['Number of cats in the household'].apply(map_cats)
    save_to_excel(df)


# 11th
def transform_age_column(df):
    df['Age'] = df['Age'].apply(map_age)
    save_to_excel(df)


# 12th and final transformation
"""
Factorizes the 'Race' column to get numerical and unique values
Adds the numerical codes as a new column called 'Numerical Race'
Renames the original 'Race' column to 'Race_Description' and keeps it
Drops the original 'Race' column
Reorders the columns so that 'Numerical Race' and 'Race_Description' are at the end
"""


def transform_race_column(df):
    # if the transformation has already been applied by checking the existence of both columns
    if 'Numerical Race' in df.columns and 'Race Description' in df.columns:
        print("Transformation has already been applied. No further changes made.")
        return df  # Return the DataFrame as is

    # factorize 'Race' column to get numerical encoding and unique race names
    numerical_race, unique_races = pd.factorize(df['Race'])

    # add 'Numerical Race' column
    df['Numerical Race'] = numerical_race

    # add 'Race Description' column and copy 'Race' values to it
    df['Race Description'] = df['Race']

    # drop the original 'Race' column
    df = df.drop(columns=['Race'])

    # reorder columns, placing 'Numerical Race', 'Race Description', 'Plus' at the end
    end_columns = ['Plus', 'Numerical Race', 'Race Description']
    other_columns = [col for col in df.columns if col not in end_columns]
    # reorder the DataFrame
    df = df[other_columns + end_columns]

    print("Race column successfully transformed.")
    return df  # Return the modified DataFrame


# the complete 'script' function
def clean_data(original_data=path):
    print_fixed_length_message('Starting the script that cleans the data')

    print_fixed_length_message('Removing the redundant columns')
    df = remove_columns(original_data)

    print_fixed_length_message("Identifying missing or 'Unknown' values")
    analyze_missing_values(df)

    print_fixed_length_message('Checking for repeated instances')
    check_repeated_instances(df)

    print_fixed_length_message('Dropping repeated instances')
    df = check_and_drop_repeated_instances(df)

    print_fixed_length_message("Modifying 'Abundance of natural areas' column")
    transform_abundance_column(df)

    print_fixed_length_message('Dropping unknown instances')
    df = check_and_drop_unknown_instances(df)

    print_fixed_length_message('Starting to convert entries to a numerical representation')

    print_fixed_length_message("Converting 'Place of living'")
    transform_place_of_living(df)

    print_fixed_length_message("Converting 'Sex' ")
    transform_sex_column(df)

    print_fixed_length_message("Converting 'Area'")
    transform_area_column(df)

    print_fixed_length_message("Converting 'Number of cats'")
    transform_number_of_cats_column(df)

    print_fixed_length_message("Converting 'Age'")
    transform_age_column(df)

    print_fixed_length_message("Finally, converting 'Race' to a numerical representation")
    transform_race_column(df)
    df = df.drop(columns=['Race'])

    print_fixed_length_message("Getting the cleaned data for training")
    get_train_data()

    print_fixed_length_message("The script that cleans the dats is officially Done")

    return df


def get_train_data():
    if os.path.exists(train_data):
        df = pd.read_excel(train_data)
        print("Training data loaded from existing file.")
        return df

    df = pd.read_excel(new_data)

    # drop the 'Plus' column
    df = df.drop(columns=['Plus'])

    df.to_excel(train_data, index=False)
    print(f"Training data saved to {train_data}")
    return df

def add_synthetic_data():
    df = pd.read_excel(train_data).drop(columns=['Race Description'])

    # separate features and target labels
    X = df.drop(columns=['Numerical Race'])
    y = df['Numerical Race']

    # target count for each class
    target_count = 1020

    # the sampling strategy, setting each class to the target count
    sampling_strategy = {label: target_count for label in y.unique()}

    # initialize and apply SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    breed_mapping = {
        0: 'Birman', 1: 'European', 2: 'No breed', 3: 'Maine coon', 4: 'Bengal',
        5: 'Persian', 6: 'Oriental', 7: 'British Shorthair', 8: 'Other',
        9: 'Chartreux', 10: 'Ragdoll', 11: 'Turkish angora', 12: 'Sphynx', 13: 'Savannah'
    }

    # Map the resampled numerical labels back to breed names
    race_description_resampled = [breed_mapping[label] for label in y_resampled]

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['Numerical Race'] = y_resampled
    df_resampled['Race Description'] = race_description_resampled

    # save and shuffle the resulting DataFrame
    df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)
    df_resampled.to_excel("balanced_train_data.xlsx", index=False)

    print(df_resampled['Numerical Race'].value_counts())

def load_data(file_path):

    df = pd.read_excel(file_path)

    # aditional check
    if df[['Numerical Race', 'Race Description']].isnull().any().any():
        print("Warning: Missing values found in 'Numerical Race' or 'Race Description' columns.")

    # randomization
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total instances in data: {len(df)}")

    # separate features and target variables
    if 'Numerical Race' in df.columns and 'Race Description' in df.columns:
        X = df.drop(columns=['Numerical Race', 'Race Description'])
        y = df['Numerical Race']
        race_desc = df['Race Description']
    else:
        raise ValueError("Columns 'Numerical Race' or 'Race Description' not found in data.")

    print("Data shape (X):", X.shape)
    print("Data shape (y):", y.shape)

    return X, y, race_desc

def split_data(X, y, method="train_test", num_folds=5):
    if method == "train_test":
        # 80-20 split
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        return X_train, y_train, X_val, y_val

    elif method == "cross_validation":
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        return kf.split(X)

    else:
        raise ValueError("Invalid splitting method. Choose 'train_test' or 'cross_validation'.")
