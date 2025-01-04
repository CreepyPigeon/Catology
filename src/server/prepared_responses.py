import os
from typing import List, Union

import numpy as np

from src.agent.custom_network.architecture import NeuralNetwork
from src.agent.custom_network.train import import_hyperparameters
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
train_dataset = os.getenv('BALANCED_DATASET')
new_excel = os.getenv('NEW_EXCEL')
saved_weights = os.getenv('TRAINED_WEIGHTS')
hyperparameters_path = '../agent/hyperparameters.yaml'

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

def convert_attributes(input_attributes: List[str]) -> Union[List[float], int]:

    """
    :param input_attributes: the raw attributes, some of them are categorical, some numerical
    :return: a numerical (if applicable) list of all the attributes
    """

    converted_attributes = []

    for index, value in enumerate(input_attributes):

        try:
            if index == 0:
                new_attribute = 1 if value == "M" else 0
            elif index == 3:
                if value == "AAB":
                    new_attribute = 1
                elif value == "ML":
                    new_attribute = 2
                elif value == "MI":
                    new_attribute = 3
                else:
                    new_attribute = 0  # default or unknown value
            elif index == 4:
                if value == 'R':
                    new_attribute = 1
                elif value == 'PU':
                    new_attribute = 2
                else:
                    new_attribute = 0
            else:  # numerical attributes
                new_attribute = float(value)  # try converting to float
                if new_attribute < 0:  # check for invalid value
                    return -1
            converted_attributes.append(new_attribute)
        except ValueError:
            return -1

    numerical_race = 0
    race_description = ''

    """
    adding the last 2 columns in order to prepare the list for addition in the dataframe
    """

    if len(input_attributes) > 26:
        if input_attributes[25] == 'Birman':
            numerical_race = 0
            race_description = 'Birman'
        if input_attributes[25] == 'European':
            numerical_race = 1
            race_description = 'European'
        if input_attributes[25] == 'No breed':
            numerical_race = 2
            race_description = 'No breed'
        if input_attributes[25] == 'Maine coon':
            numerical_race = 3
            race_description = 'Maine coon'
        if input_attributes[25] == 'Bengal':
            numerical_race = 4
            race_description = 'Bengal'
        if input_attributes[25] == 'Persian':
            numerical_race = 5
            race_description = 'Persian'
        if input_attributes[25] == 'Oriental':
            numerical_race = 6
            race_description = 'Oriental'
        if input_attributes[25] == 'British Shorthair':
            numerical_race = 7
            race_description = 'British Shorthair'
        if input_attributes[25] == 'Other':
            numerical_race = 8
            race_description = 'Other'
        if input_attributes[25] == 'Chartreux':
            numerical_race = 9
            race_description = 'Chartreux'
        if input_attributes[25] == 'Ragdoll':
            numerical_race = 10
            race_description = 'Ragdoll'
        if input_attributes[25] == 'Turkish angora':
            numerical_race = 11
            race_description = 'Turkish angora'
        if input_attributes[25] == 'Sphynx':
            numerical_race = 12
            race_description = 'Sphynx'
        if input_attributes[25] == 'Savannah':
            numerical_race = 13
            race_description = 'Savannah'

        converted_attributes.append(numerical_race)
        converted_attributes.append(race_description)
    else:
        converted_attributes.append('Unknown')
        converted_attributes.append('Unknown')

    return converted_attributes

def add_new_instance(input_attributes: List[str], file_path=train_dataset, new_file_path=new_excel):

    df = pd.read_excel(file_path)
    new_row = convert_attributes(input_attributes)

    if isinstance(new_row, int):
        return -1
    _, input_size, _, _, _, _ = import_hyperparameters(hyperparameters_path)

    if input_size == len(input_attributes) or input_size == len(input_attributes) - 1:
        df.loc[len(df)] = new_row
        df.to_excel(new_file_path, index=False)
        print('New row added successfully')
        return 1
    else:
        return -1

def classify_instance(attributes, stored_weights=saved_weights):

    i_epochs, i_input_size, i_hidden_size, i_output_size, i_learning_rate, i_batch_size = import_hyperparameters()

    nn = NeuralNetwork(input_size=i_input_size, hidden_size=i_hidden_size, output_size=i_output_size,
                       learning_rate=i_learning_rate)

    nn.load_weights(stored_weights)
    X_input = attributes.reshape(1, -1)  # reshape to a 2D array (batch size of 1)

    raw_output = nn.forward(X_input)

    print(f"Raw output(before softmax) for the input: {raw_output}")

    prediction = np.argmax(raw_output, axis=1)

    return prediction


if __name__ == '__main__':
    x_in = ["F", 2, 3, "MI", "PU", 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 3, 4]
    add_new_instance(x_in)
