import os
import pandas as pd


def build_comparison_list(file_path, first_race: str, second_race: str, num_instances: int = 100) -> tuple | None:
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

    if len(first_instance_list) != len(second_instance_list):
        print(f"The first instance list {first_instance_list} doesn't have the same length as the second instance list "
              f"{second_instance_list}")

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


def compare_races(file_path, first_race: str, second_race: str, num_instances: int = 100) -> str:
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
        the_other_value = 100 - value

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


if __name__ == '__main__':
    compare_races("balanced_train_data.xlsx", 'Persian', 'Bengal')
