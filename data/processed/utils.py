import os
from dotenv import load_dotenv

load_dotenv()
errors_file = os.getenv('ERRORS_FILE')
new_data = os.getenv('NEW_DATA')

def write_errors(content):
    with open(errors_file, 'a') as f:
        f.write(content + '\n')

def save_to_excel(data, file_name=new_data):
    # data MUST be a dataframe
    data.to_excel(file_name, index=False)

def print_fixed_length_message(message, total_length=63):
    dots_needed = total_length - len(message)
    dots = '.' * dots_needed
    print(f"{message}{dots}")

def map_age(age: float):
    if isinstance(age, (int, float)):
        return age

    if age == 'Less than 1':
        return 0.5
    elif age == '1-2 years':
        return 1.5
    elif age == '2-10 years':
        return 6
    elif age == 'More than 10':
        return 15

def map_area(designated_area):
    if isinstance(designated_area, int):
        return designated_area

    if designated_area == 'U':
        return 1
    elif designated_area == 'R':
        return 2
    elif designated_area == 'PU':
        return 3

def map_cats(number_of_cats):
    return 6 if number_of_cats == 'Over 5' else int(number_of_cats)

def map_sex(sex):
    if sex == 'F':
        return 0
    elif sex == 'M':
        return 1

    return sex

def map_place_of_living(place):
    if place == "ASB":
        return 1
    elif place == "AAB":
        return 2
    elif place == "ML":
        return 3
    elif place == "MI":
        return 4
    else:
        return place
