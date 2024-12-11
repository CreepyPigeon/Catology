import nltk
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from langdetect import detect, detect_langs

from collections import Counter
import string

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)


def get_hypernyms(word):
    hypernyms = set()
    for syn in wordnet.synsets(word):
        for hypernym in syn.hypernyms():
            hypernyms.update([lemma.name().replace('_', ' ') for lemma in hypernym.lemmas()])
    return list(hypernyms)


def get_negated_antonym(word):
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                return f"not {lemma.antonyms()[0].name().replace('_', ' ')}"
    return None


def generate_alternative_text(text, replacement_ratio=0.2, new_filename='alternative.txt'):
    words = word_tokenize(text)
    altered_words = []
    num_replacements = max(1, int(len(words) * replacement_ratio))

    for word in words:
        # Skip non-alphabetical tokens
        if not word.isalpha():
            altered_words.append(word)
            continue

        # Decide whether to replace the word
        if num_replacements > 0 and random.random() < replacement_ratio:
            synonyms = get_synonyms(word)
            hypernyms = get_hypernyms(word)
            negated_antonym = get_negated_antonym(word)

            # Create a pool of alternatives
            alternatives = synonyms + hypernyms
            if negated_antonym:
                alternatives.append(negated_antonym)

            if alternatives:
                altered_words.append(random.choice(alternatives))
                num_replacements -= 1
            else:
                altered_words.append(word)
        else:
            altered_words.append(word)

    new_text = ' '.join(altered_words)

    try:
        with open(new_filename, 'w') as f:
            f.write(new_text)
        print(f"Generated alternative text successfully to {new_filename}")
    except IOError:
        print('Error writing to file')

    return new_text

def stylometric_analysis(text):
    # Tokenize words
    words = nltk.word_tokenize(text)
    # Filter out punctuation
    words = [word for word in words if word.isalpha()]
    # Tokenize sentences
    sentences = nltk.sent_tokenize(text)

    # Calculate metrics
    num_words = len(words)
    num_sentences = len(sentences)
    num_characters = sum(len(word) for word in words)
    avg_word_length = num_characters / num_words if num_words > 0 else 0
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

    # Word frequencies
    word_frequencies = Counter(words)

    # Print results
    print("Stylometric Analysis:")
    print(f"Total words: {num_words}")
    print(f"Total characters (excluding spaces): {num_characters}")
    print(f"Average word length: {avg_word_length:.2f}")
    print(f"Average sentence length (in words): {avg_sentence_length:.2f}")
    print(f"Word frequencies: {word_frequencies.most_common(10)}")

def read_content(filename, read_from_console=False):

    if read_from_console:
        text = input('Enter some text here: ')
        print(text)

        return text

    try:
        with open(filename, "r") as f:
            content= f.read()
            print(content)

        return content
    except FileNotFoundError:
        print("File not found")

if __name__ == '__main__':
    filename = 'input_file.txt'
    contents = read_content(filename, False)

    language = detect(contents)
    print(f"Detected language: {language}")

    probabilities = detect_langs(contents)
    print(f"Language probabilities: {probabilities}")

    stylometric_analysis(contents)

    generate_alternative_text(contents)

