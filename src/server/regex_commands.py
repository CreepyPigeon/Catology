import re

def fun():
    pattern = """add a new instance in the cat dataset. it is of male, age 4, place of living: rural area, outdoors time: medium, 
                 highly predictable, well-above friendly, medium distracted, not so much afraid, and has highly bird capturing frequency."""

    # Initialize the mapping dictionary
    attribute_mapping = []

    # Regex to capture the attributes and values

    # as putea sa fac asta pentru fiecare coloana in parte
    regex_patterns = [
        (r"\bmale\b", "Sex"),  # Matches 'male'
        (r"\bage (\d+)\b", "Age"),  # Matches 'age 4'
        (r"\bplace of living: (rural area|urban area|periurban area)\b", "Place of living"),  # Matches place of living
        (r"\boutdoors time: (medium|highly|well-above|not so much)\b", "Outdoors time"),  # Matches outdoors time
        (r"\bhighly (predictable)\b", "Predictable"),  # Matches highly predictable
        (r"\bwell-above (friendly)\b", "Friendly"),  # Matches well-above friendly
        (r"\bmedium (distracted)\b", "Distracted"),  # Matches medium distracted
        (r"\bnot so much (afraid)\b", "Afraid"),  # Matches not so much afraid
        (r"\bhas highly (bird capturing frequency)\b", "Bird capturing frequency"),  # Matches bird capturing frequency
    ]

    # Loop through patterns and extract matches
    for regex, column in regex_patterns:
        match = re.search(regex, pattern)
        if match:
            value = match.group(1) if match.lastindex else match.group(0)
            attribute_mapping.append({column: value})

    # Print the result
    print("Extracted Attributes:", attribute_mapping)


def process_attribute_mapping(attribute_mapping):
    # Initialize the result list
    processed_output = []

    # Iterate over the dictionary
    for column, entries in attribute_mapping.items():
        for entry in entries:
            # Create a dictionary for each column-value pair
            processed_output.append({column: entry['keyword']})

    return processed_output


if __name__ == '__main__':

    current_mapping = fun()
    # output: Extracted Attributes: [{'Sex': 'male'}, {'Age': '4'}, {'Place of living': 'rural area'},
    # {'Outdoors time': 'medium'}, {'Predictable': 'predictable'}, {'Friendly': 'friendly'}, {'Distracted': 'distracted'}, {'Afraid': 'afraid'}, {'Bird capturing frequency': 'bird capturing frequency'}]

    # processed_output = process_attribute_mapping(current_mapping)
    # print("Processed Output:", processed_output)
