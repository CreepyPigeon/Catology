import os
import random

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import List, Union, Dict
from src.agent.custom_network.architecture import NeuralNetwork
from src.agent.custom_network.train import import_hyperparameters
from transformers import AutoModelForCausalLM, AutoTokenizer


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

def get_value_from_dictionary(keyword: str, column_name: str):
    """
    get a value from the specified column based on the keyword and proportional to the max value of the column.

    :param keyword: a string that defines the threshold ('highly', 'high', 'well-above', 'medium',
                    'not so', 'not so much')
    :param column_name: The column name from which the value is retrieved
    :param dataset: The DataFrame containing the data
    :return: The value corresponding to the specified keyword
    """

    column_values = data[column_name]

    # thresholds
    max_value = column_values.max()
    min_value = column_values.min()
    mean_value = column_values.mean()
    sixth_of_max = max_value / 6

    if keyword == 'highly' or keyword == 'high':
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
    elif keyword == 'not so much' or keyword == 'not so':
        # bottom 1/6 of the values
        threshold = min_value + sixth_of_max
        result = column_values[column_values <= threshold].min()
    else:
        print("Keyword not recognized! Use 'highly', 'high, 'well-above', 'medium', or 'not so', 'not so much.")
        return -1

    return result

def get_numerical_mapping(mapping, dataset=data):
    numerical_mapping = []
    for item in mapping:
        for column_name, keyword in item.items():
            if column_name == "Sex":
                # special case for Sex
                if keyword == "male":
                    numerical_mapping.append({column_name: 1})
                elif keyword == "female":
                    numerical_mapping.append({column_name: 0})
                else:
                    print(f"Unknown value for Sex: {keyword}")
            elif column_name == "Age":
                # age is numeric, so keep it as is
                numerical_mapping.append({column_name: int(keyword)})
            elif column_name in dataset.columns:
                # Use the existing function for other attributes
                numerical_value = get_value_from_dictionary(keyword, column_name)
                numerical_mapping.append({column_name: numerical_value})
            else:
                return -1
    return numerical_mapping


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

    columns1 = df['Race Description'].str.lower() == first_race.lower()
    columns2 = df['Race Description'].str.lower() == second_race.lower()

    if not columns1.any() or not columns2.any():
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
        return "One or both of the specified cat breeds doesn't exist"
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
    return random_slice_response(result)

def random_slice_response(response, min_phrases=3):
    # split into phrases
    phrases = response.split(". ")
    # remove any empty strings (if present)
    phrases = [phrase for phrase in phrases if phrase.strip()]
    # shuffle the phrases
    random.shuffle(phrases)
    # at least `min_phrases` are included
    selected_phrases = phrases[:max(min_phrases, random.randint(3, len(phrases)))]
    return ". ".join(selected_phrases) + "."

def complete_attributes(input_attributes: List[Dict[str, int]]):

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
        _, race_description = add_label(converted_attributes)

    return converted_attributes, race_description

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
    return attributes, race_description

def add_new_instance(column_names: List[Dict[str, int]], add, file_path=train_dataset, new_file_path=new_excel):

    """
    :param add: boolean paramater, whether or not we should add the instance to the dataset, or just classify it
    :param new_file_path:
    :param file_path:
    :param column_names: the input column names, given as a list of key-value pairs (column_name:value)
     got after a pre-processing step
    """

    df = pd.read_excel(file_path)
    new_row, label = complete_attributes(column_names)
    print(f"new_row = {new_row}")

    if add:
        df.loc[len(df)] = new_row
        df.to_excel(new_file_path, index=False)
        print('New row added successfully')

    return label

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

def describe_cat_breed(cat_breed, file_path=train_dataset):
    """
    :param cat_breed: The breed to be described. Capitalization doesn`t matter.
    :param file_path: Path to the Excel dataset.
    :return: Natural language description of a cat breed.
    """

    df = pd.read_excel(file_path)
    breed_data = df[df['Race Description'].str.contains(cat_breed, case=False, na=False)]
    significant_columns = []
    result = f"The breed '{cat_breed}' has these characteristics:\n"

    for column in df.columns:
        if column == "Numerical Race":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            value_counts = breed_data[column].value_counts(normalize=True)
            if not value_counts.empty:
                most_frequent_value = value_counts.index[0]
                proportion = value_counts.iloc[0]
                if proportion >= 0.7:
                    significant_columns.append((column, most_frequent_value, proportion))

    if significant_columns:
        for column, value, proportion in significant_columns:
            if proportion < 0.9:
                result += "Most cats "
            else:
                result += "The vast majority of the cats "

            if column == 'Sex':
                if value == 0:
                    result += "are female.\n"
                else:
                    result += "are male.\n"
            elif column == 'Age':
                if value < 1:
                    result += "are under one year old.\n"
                elif value < 10:
                    result += "are between 2 and 10 years old.\n"
                else:
                    result += "are over 10 years old.\n"
            elif column == 'Number of cats in the household':
                if value < 6:
                    result += f"have {value} cats in the household.\n"
                else:
                    result += "have more than 5 cats in the household.\n"
            elif column == 'Place of living':
                options = {
                    0: "live in an apartment without a balcony.",
                    1: "live in an apartment with a balcony or terrace.",
                    2: "live in a house in a subdivision.",
                    3: "live in an individual house.",
                }
                result += options.get(value, "") + "\n"
            elif column == 'Urban/Rural area':
                options = {
                    0: "live in an urban area.",
                    1: "live in a rural area.",
                    2: "live in a periurban area.",
                }
                result += options.get(value, "") + "\n"
            elif column == 'Outdoors time':
                options = {
                    0: "spend no time outdoors.",
                    1: "spend little time outdoors.",
                    2: "spend moderate time outdoors.",
                    3: "spend long time outdoors.",
                    4: "live outdoors.",
                }
                result += options.get(value, "") + "\n"
            elif column == 'Time spent with cat':
                options = {
                    0: "spend no time with their owners.",
                    1: "spend little time with their owners.",
                    2: "spend moderate time with their owners.",
                    3: "spend lots of time with their owners.",
                }
                result += options.get(value, "") + "\n"
            elif column in ['Timid', 'Calm', 'Afraid', 'Intelligent', 'Vigilant', 'Persevering', 'Affectionate', 'Friendly', 'Lonely', 'Brutal', 'Dominant', 'Aggressive', 'Impulsive', 'Predictable', 'Distracted']:
                intensity = {
                    1: "not ",
                    2: "a bit ",
                    3: "moderately ",
                    4: "pretty ",
                    5: "very ",
                }
                result += f"are {intensity.get(value, '')}{column}.\n"
            elif column == 'Abundance of natural areas':
                options = {
                    1: "live in a place with few natural areas.",
                    2: "live in a place with a moderate amount of natural areas.",
                    3: "live in a place with lots of natural areas.",
                }
                result += options.get(value, "") + "\n"
            elif column == 'Bird capturing frequency':
                options = {
                    0: "never capture birds.",
                    1: "rarely capture birds.",
                    2: "capture birds sometimes.",
                    3: "capture birds often.",
                    4: "capture birds very often.",
                }
                result += options.get(value, "") + "\n"
            elif column == 'Mammal capturing frequency':
                options = {
                    0: "never capture mammals.",
                    1: "rarely capture mammals.",
                    2: "capture mammals sometimes.",
                    3: "capture mammals often.",
                    4: "capture mammals very often.",
                }
                result += options.get(value, "") + "\n"

    else:
        result += f"This cat breed is perfectly ordinary: '{cat_breed}'."

    return result

def generate_gpt_response(message):
    checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    messages = [{"role": "system", "content": "You are a chatbot that talks about cats. Please answer all questions with a single short sentence. No explanations, just the answer."}, {"role": "user", "content": message}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    #print(input_text)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=64, temperature=0.1,top_p=0.9, do_sample=True, early_stopping=True)
    #print(tokenizer.decode(outputs[0]))
    response = tokenizer.decode(outputs[0])
    return response

def clean_gpt_response(response: str) -> str:
    """
    Cleans the response by removing tags and returning only the content after <|im_start|>assistant.
    :param response: The raw response string.
    :return: The cleaned response string.
    """
    # Split the response by '<|im_start|>' and take the content after the first occurrence.
    parts = response.split('<|im_start|>assistant')

    # If there's content after '<|im_start|>', return it; otherwise, return the original response.
    if len(parts) > 1:
        return parts[1].split('<|im_end|>')[0].strip()
    return response  # In case there's no '<|im_start|>' tag, return the original response


if __name__ == '__main__':
    #input_dict = [{'Lonely': 3}, {'Brutal': 4}, {'Dominant':2}, {'Aggressive':1}]
    #add_new_instance(input_dict)

    description_of_Sphynx = describe_cat_breed("Sphynx", train_dataset)
    print(description_of_Sphynx)