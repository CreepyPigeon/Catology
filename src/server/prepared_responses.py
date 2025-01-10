import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import List, Union, Dict
from src.agent.custom_network.architecture import NeuralNetwork
from src.agent.custom_network.train import import_hyperparameters


load_dotenv()
train_dataset = os.getenv('BALANCED_DATASET')
new_excel = os.getenv('NEW_EXCEL')
saved_weights = os.getenv('TRAINED_WEIGHTS')
hyperparameters_path = '../agent/hyperparameters.yaml'

"""
define the column means as global variables, since they fill be frequently needed (exclude the last one, since it is
non numeric
"""
data = pd.read_excel(train_dataset)
numeric_columns = data.select_dtypes(include=['number'])
column_means = numeric_columns.mean().to_dict()
COLUMN_MEANS = column_means

def get_column_means():
    return COLUMN_MEANS

def get_value_from_dictionary(keyword: str, column_name: str, dataset: pd.DataFrame):
    """
    get a value from the specified column based on the keyword and proportional to the max value of the column.

    :param keyword: a string that defines the threshold ('highly', 'well-above', 'medium', 'not so')
    :param column_name: The column name from which the value is retrieved
    :param dataset: The DataFrame containing the data
    :return: The value corresponding to the specified keyword
    """

    column_values = dataset[column_name]

    # thresholds
    max_value = column_values.max()
    min_value = column_values.min()
    mean_value = column_values.mean()
    sixth_of_max = max_value / 6

    if keyword == 'highly':
        # top 1/6th of the max value
        threshold = max_value - sixth_of_max
        result = column_values[column_values >= threshold].max()
    elif keyword == 'well-above':
        # top 1/6 after the max value
        threshold = max_value - 2 * sixth_of_max
        result = column_values[column_values >= threshold].max()
    elif keyword == 'medium':
        #  the mean value for that column
        result = mean_value
    elif keyword == 'not so much':
        # bottom 1/6 of the values
        threshold = min_value + sixth_of_max
        result = column_values[column_values <= threshold].min()
    else:
        print("Keyword not recognized! Use 'highly', 'well-above', 'medium', or 'not so'.")
        return -1

    return result

def build_comparison_list(file_path, first_race: str, second_race: str, num_instances: int = 500) -> tuple | None:
    """
    the idea is to sample from the file_path argument a fixed number of instances and compare their attributes directly,
    meaning which instance has the bigger numerical value

    based on this, we can build a comparison list, containing the number of instances where the second race has the
    bigger value
    :return: the name of the columns, as well as the comparison list
    """

    df = pd.read_excel(file_path)

    if df is None:
        return None

    columns1 = df['Race Description'] == first_race
    columns2 = df['Race Description'] == second_race

    if columns1 is None or columns2 is None:
        return None

    first_instance_list = df[columns1]
    random_instances1 = first_instance_list.sample(n=num_instances).iloc[:, :-1]  # without the last column

    second_instance_list = df[columns2]
    random_instances2 = second_instance_list.sample(n=num_instances).iloc[:, :-1]

    if random_instances1.shape != random_instances2.shape:
        print(f"Cat breeds sampled from {first_race} have shape {random_instances1.shape} while the second "
              f"instance list {second_race} have shape {random_instances2.shape}")

    column_names = list(df.columns[:-2])

    comparison_list = [0 for _ in range(len(column_names))]
    for i in range(len(column_names)):

        count = 0
        current_column = column_names[i]
        for j in range(len(random_instances1)):

            instance1 = random_instances1.iloc[j]
            instance2 = random_instances2.iloc[j]
            if instance1[current_column] < instance2[current_column]:
                count += 1

        comparison_list[i] = count

    return column_names, comparison_list

def compare_races(first_race: str, second_race: str, num_instances: int = 500, file_path=train_dataset) -> str:

    result = build_comparison_list(file_path, first_race, second_race, num_instances)

    if result is None:
        return "One or both of the specified cat breeds don't exist"
    else:
        column_names, comparison_list = result

    response_template = ("The {cat_race1} race is {number_of_times1} more likely to be a male. "
                         "The {cat_race2} race is {number_of_times2} more likely to be older. "
                         "The {cat_race3} race is {number_of_times3} more likely to live in a rural area. "
                         "The {cat_race4} race is {number_of_times4} more timid. "
                         "The {cat_race5} race is {number_of_times5} more intelligent. "
                         "The {cat_race6} race is {number_of_times6} more friendly. "
                         "The {cat_race7} race is {number_of_times7} more affectionate. "
                         "The {cat_race8} race is {number_of_times8} more predictable. "
                         "The {cat_race9} race has a bird capturing frequency {number_of_times9} higher.")

    formatted_parts = {}
    for i in range(len(comparison_list)):

        value = comparison_list[i]
        the_other_value = num_instances - value

        cat_race = first_race if the_other_value > value else second_race
        percentage = the_other_value / value if the_other_value > value else value / the_other_value
        percentage = round(percentage, 2)

        if column_names[i] == 'Sex':
            formatted_parts['cat_race1'] = cat_race
            formatted_parts['number_of_times1'] = percentage
        elif column_names[i] == 'Age':
            formatted_parts['cat_race2'] = cat_race
            formatted_parts['number_of_times2'] = percentage
        elif column_names[i] == 'Urban/Rural area':
            formatted_parts['cat_race3'] = cat_race
            formatted_parts['number_of_times3'] = percentage
        elif column_names[i] == 'Timid':
            formatted_parts['cat_race4'] = cat_race
            formatted_parts['number_of_times4'] = percentage
        elif column_names[i] == 'Intelligent':
            formatted_parts['cat_race5'] = cat_race
            formatted_parts['number_of_times5'] = percentage
        elif column_names[i] == 'Friendly':
            formatted_parts['cat_race6'] = cat_race
            formatted_parts['number_of_times6'] = percentage
        elif column_names[i] == 'Affectionate':
            formatted_parts['cat_race7'] = cat_race
            formatted_parts['number_of_times7'] = percentage
        elif column_names[i] == 'Predictable':
            formatted_parts['cat_race8'] = cat_race
            formatted_parts['number_of_times8'] = percentage
        elif column_names[i] == 'Bird capturing frequency':
            formatted_parts['cat_race9'] = cat_race
            formatted_parts['number_of_times9'] = percentage

    result = response_template.format(**formatted_parts)
    print(result)

def complete_attributes(input_attributes: List[Dict[str, int]]) -> Union[List[float], int]:

    """
    completes missing attributes in input data using precomputed column means.

    :param input_attributes: a list of dictionaries with specified attributes.
    :return: a numerical list of all attributes, with missing values filled by column means.
    """
    # the order of columns in the dataset
    columns_in_order = [
        "Sex", "Age", "Number of cats in the household", "Place of living",
        "Urban/Rural area", "Outdoors time", "Time spent with cat", "Timid",
        "Calm", "Afraid", "Intelligent", "Vigilant", "Persevering", "Affectionate",
        "Friendly", "Lonely", "Brutal", "Dominant", "Aggressive", "Impulsive",
        "Predictable", "Distracted", "Abundance of natural areas",
        "Bird capturing frequency", "Mammal capturing frequency", "Numerical Race",
        "Race Description"
    ]

    converted_attributes = []

    flattened_attributes = {}
    for attr in input_attributes:
        flattened_attributes.update(attr)

    for column in columns_in_order:
        if column == 'Race Description' or column == 'Numerical Race':
            # skip Race Description column
            continue

        if column in flattened_attributes:
            # use the value from input_attributes if present
            converted_attributes.append(flattened_attributes[column])
        else:
            # otherwise, use the mean value
            column_mean_values = get_column_means()
            if column in column_means:
                converted_attributes.append(column_mean_values[column])
            else:
                # no mean is available, should be impossible
                converted_attributes.append(None)

    """
    adding the last 2 columns in order to prepare the list for addition in the dataframe
    """

    # if the input contains it, add it
    numerical_race = 0
    race_description = ''
    found_race = False
    for attr in input_attributes:
        if 'Race Description' in attr:
            if 'Race Description' == 'Birman':
                numerical_race = 0
                race_description = 'Birman'
            if 'Race Description' == 'European':
                numerical_race = 1
                race_description = 'European'
            if 'Race Description' == 'No breed':
                numerical_race = 2
                race_description = 'No breed'
            if 'Race Description' == 'Maine coon':
                numerical_race = 3
                race_description = 'Maine coon'
            if 'Race Description' == 'Bengal':
                numerical_race = 4
                race_description = 'Bengal'
            if 'Race Description' == 'Persian':
                numerical_race = 5
                race_description = 'Persian'
            if 'Race Description' == 'Oriental':
                numerical_race = 6
                race_description = 'Oriental'
            if 'Race Description' == 'British Shorthair':
                numerical_race = 7
                race_description = 'British Shorthair'
            if 'Race Description' == 'Other':
                numerical_race = 8
                race_description = 'Other'
            if 'Race Description' == 'Chartreux':
                numerical_race = 9
                race_description = 'Chartreux'
            if 'Race Description' == 'Ragdoll':
                numerical_race = 10
                race_description = 'Ragdoll'
            if 'Race Description' == 'Turkish angora':
                numerical_race = 11
                race_description = 'Turkish angora'
            if 'Race Description' == 'Sphynx':
                numerical_race = 12
                race_description = 'Sphynx'
            if 'Race Description' == 'Savannah':
                numerical_race = 13
                race_description = 'Savannah'

            found_race = True
            converted_attributes.append(numerical_race)
            converted_attributes.append(race_description)

    print(f"converted_attributes before classification {converted_attributes}")
    # classify the instance and append its most probable class
    if not found_race:
        add_label(converted_attributes)

    return converted_attributes

def add_label(attributes):

    """
    :return: return the most probable class after classification
    """

    race_description = ''
    label = classify_instance(np.array(attributes))

    if label == 0:
        race_description = 'Birman'
    if label == 1:
        race_description = 'European'
    if label == 2:
        race_description = 'No breed'
    if label == 3:
        race_description = 'Maine coon'
    if label == 4:
        race_description = 'Bengal'
    if label == 5:
        race_description = 'Persian'
    if label == 6:
        race_description = 'Oriental'
    if label == 7:
        race_description = 'British Shorthair'
    if label == 8:
        race_description = 'Other'
    if label == 9:
        race_description = 'Chartreux'
    if label == 10:
        race_description = 'Ragdoll'
    if label == 11:
        race_description = 'Turkish angora'
    if label == 12:
        race_description = 'Sphynx'
    if label == 13:
        race_description = 'Savannah'

    attributes.append(label)
    attributes.append(race_description)
    print(f"Most probable class {race_description}")
    return attributes

def add_new_instance(column_names: List[Dict[str, int]], file_path=train_dataset, new_file_path=new_excel):

    """
    :param column_names: the input column names, given as a list of key-value pairs (column_name:value)
     got after a pre-processing step
    :param file_path:
    :param new_file_path:
    :return:
    """

    df = pd.read_excel(file_path)
    new_row = complete_attributes(column_names)

    df.loc[len(df)] = new_row
    df.to_excel(new_file_path, index=False)
    print('New row added successfully')

    return 1

def classify_instance(attributes, stored_weights=saved_weights, file=hyperparameters_path):

    i_epochs, i_input_size, i_hidden_size, i_output_size, i_learning_rate, i_batch_size = import_hyperparameters(file)

    nn = NeuralNetwork(input_size=i_input_size, hidden_size=i_hidden_size, output_size=i_output_size,
                       learning_rate=i_learning_rate)

    nn.load_weights(stored_weights)
    X_input = attributes.reshape(1, -1)  # reshape to a 2D array (batch size of 1)

    raw_output = nn.forward(X_input)

    print(f"Raw output(before softmax) for the input: {raw_output}")

    prediction = np.argmax(raw_output, axis=1)

    return prediction


if __name__ == '__main__':
    input_dict = [{'Lonely': 3}, {'Brutal': 4}, {'Dominant':2}, {'Aggressive':1}]
    add_new_instance(input_dict)
