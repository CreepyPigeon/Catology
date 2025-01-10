import re

def fun():
    pattern = """add a new instance in the cat dataset. it is male, age 4, living in a rural area, spends medium time outside, 
                 highly predictable, well-above friendly, medium distracted, not so afraid, and has highly mammal capturing frequency."""

    # Define value keywords and their mappings to column names
    column_mapping = {
        'male': 'Sex',
        'female': 'Sex',
        'age': 'Age',
        'living in a rural area': 'Place of living',
        'living in a urban area': 'Place of living',
        'living in a periurban area': 'Place of living',
        'spends': 'Outdoors time',
        'highly': 'Predictable',  # 'highly' maps to 'Predictable' column
        'well-above': 'Friendly',  # 'well-above' maps to 'Friendly'
        'medium': 'Distracted',  # 'medium' maps to 'Distracted'
        'not so': 'Afraid',  # 'not so' maps to 'Afraid'
        'has': 'Mammal capturing frequency'  # 'has' maps to 'Mammal capturing frequency'
    }

    # Regex pattern to match each part of the sentence
    regex = r"(\b(?:male|female|age|living in a (?:rural|urban|periurban) area|spends|highly|well-above|medium|not so|has)\b)\s*(\w+|\d+|\w+\s\w+)"

    # Find all matches
    matches = re.findall(regex, pattern)

    # Initialize a dictionary to store the results
    attribute_mapping = {}

    # Process matches and map them
    for match in matches:
        keyword, value = match

        # Handle direct value-to-column mapping for known keywords
        if keyword in column_mapping:
            column = column_mapping[keyword]
            if column not in attribute_mapping:
                attribute_mapping[column] = []
            attribute_mapping[column].append({"keyword": keyword, "value": value})
        else:
            # For combined value keywords like 'highly predictable'
            if keyword in ['highly', 'well-above', 'medium', 'not so']:
                # Extract the appropriate column based on the first part of the value (e.g., 'predictable' for 'highly predictable')
                column = value  # Use the second word as the column
                if column not in attribute_mapping:
                    attribute_mapping[column] = []
                attribute_mapping[column].append(
                    {"keyword": keyword, "value": keyword})  # Keep the value as 'highly', 'medium', etc.

    # Print the final attribute mapping
    print("Final Attribute Mapping:", attribute_mapping)


if __name__ == '__main__':
    fun()
