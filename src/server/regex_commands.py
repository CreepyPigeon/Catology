import re

def get_mapping_instance_addition(pattern):

    attribute_mapping = []

    # patterns with options for attributes that require it
    options = {
        "Outdoors time": r"(medium|highly|high|well-above|not so much|not so)",
        "Predictable": r"(medium|highly|high|well-above|not so much|not so)",
        "Distracted": r"(medium|highly|high|well-above|not so much|not so)",
        "Numer of cats in the household": r"(medium|highly|high|well-above|not so much|not so)",
        "Time spent with cat": r"(medium|highly|high|well-above|not so much|not so)",
        "Timid": r"(medium|highly|high|well-above|not so much|not so)",
        "Calm": r"(medium|highly|high|well-above|not so much|not so)",
        "Intelligent": r"(medium|highly|high|well-above|not so much|not so)",
        "Vigilant": r"(medium|highly|high|well-above|not so much|not so)",
        "Persevering": r"(medium|highly|high|well-above|not so much|not so)",
        "Affectionate": r"(medium|highly|high|well-above|not so much|not so)",
        "Friendly": r"(medium|highly|high|well-above|not so much|not so)",
        "Lonely": r"(medium|highly|high|well-above|not so much|not so)",
        "Brutal": r"(medium|highly|high|well-above|not so much|not so)",
        "Afraid": r"(medium|highly|high|well-above|not so much|not so)",
        "Dominant": r"(medium|highly|high|well-above|not so much|not so)",
        "Aggressive": r"(medium|highly|high|well-above|not so much|not so)",
        "Impulsive": r"(medium|highly|high|well-above|not so much|not so)",
        "Abundance of natural areas": r"(medium|highly|high|well-above|not so much|not so)",
        "Mammal capturing frequency": r"(medium|highly|high|well-above|not so much|not so)",
        "Bird capturing frequency": r"(medium|highly|high|well-above|not so much|not so)"
    }

    # regex patterns for all attributes
    regex_patterns = [
        (r"\b(male|female)\b", "Sex"),  # Exact match for 'male'
        (r"\bage (\d+)\b", "Age"),  # age
        (r"\bplace of living: (rural area|urban area|periurban area)\b", "Place of living"),  # Match place of living
    ]

    # dynamic regex for options
    for column, option_pattern in options.items():
        regex_patterns.append((fr"\b{option_pattern} {column.lower()}\b", column))

    # extract matches
    for regex, column in regex_patterns:
        match = re.search(regex, pattern)
        if match:
            value = match.group(1) if match.lastindex else match.group(0)
            attribute_mapping.append({column: value})

    print("Extracted Attributes:", attribute_mapping)
    return attribute_mapping

def get_two_breeds(message):

    # potentially multiple words
    pattern = r"generate a natural language (?:description|comparison) between ([a-zA-Z\s]+) race and ([a-zA-Z\s]+) " \
              r"race"

    match = re.search(pattern, message)
    if match:
        race1 = match.group(1).strip()
        race2 = match.group(2).strip()
        print(f"Extracted races: {race1} and {race2}")
    else:
        return -1, -1

    return race1, race2


if __name__ == '__main__':

    input_pattern = """add a new instance in the cat dataset. it is a male, age 1, not so timid,
        outdoors time: well-above, not so predictable, highly calm, highly brutal, not so impulsive,
        and has highly mammal capturing frequency."""

    current_mapping = get_mapping_instance_addition(input_pattern)
