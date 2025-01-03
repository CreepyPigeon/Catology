import nltk
import random
from nltk import pos_tag
from nltk.data import find
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, detect_langs
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import matplotlib.pyplot as plt
from transformers import pipeline
import re


def download_nltk_resource(resource_name):
    try:
        find(resource_name)
    except LookupError:
        nltk.download(resource_name)


nltk.download('averaged_perceptron_tagger_eng')
download_nltk_resource('punkt')
download_nltk_resource('punkt-tab')
download_nltk_resource('wordnet')


def preprocess_text(text):
    nltk.download('stopwords')
    nltk.download('punkt')

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]  # remove punctuation
    words = [word for word in words if word not in stop_words]  # remove stopwords
    return ' '.join(words)


def extract_keywords_tfidf(text, n_keywords=5):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    keywords = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
    keywords = sorted(keywords, key=lambda x: x[1], reverse=True)

    return [keyword[0] for keyword in keywords[:n_keywords]]


def extract_keywords_rake(text, n_keywords=5):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases_with_scores()
    return [phrase[1] for phrase in keywords[:n_keywords]]


def generate_sentences_for_keywords(text, keywords):
    sentences = nltk.sent_tokenize(text)
    sentences_for_keywords = []

    for keyword in keywords:
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                sentences_for_keywords.append(f"The keyword '{keyword}' appears in the sentence: {sentence}")
                break  # stop after finding the first sentence with the keyword
        print(keyword, sentence)
        generated_by_gpt = generate_sentence_with_gpt(keyword, sentence)
        cleanup_responses_from_string(generated_by_gpt)
    return sentences_for_keywords


def generate_sentence_with_gpt(word, context):
    prompt = f"Write a sentence with the word: '{word}'. Like this: '{context}'."

    result = generator(prompt, max_length=45, num_return_sequences=1, no_repeat_ngram_size=2)
    sentence = result[0]['generated_text']
    print(sentence)
    return sentence


def cleanup_responses_from_string(input_string):
    match = re.search(r"word: '(\w+)'", input_string)
    if not match:
        raise ValueError("Could not find the target word in the input string.")
    target_word = match.group(1)

    sentence_pattern = re.compile(rf"([^.!?]*\b{re.escape(target_word)}\b[^.!?]*[.!?])", re.IGNORECASE)

    sentences = sentence_pattern.findall(input_string)

    cleaned_sentences = [sentence.strip() for sentence in sentences]
    print(f"Cleaned {cleaned_sentences[-1]}")
    return cleaned_sentences


def generate_alternative_text(text, replacement_ratio=0.2, replace_synonyms=True, replace_hypernyms=True,
                              replace_antonyms=True, new_filename='alternative.txt'):
    words = word_tokenize(text)
    altered_words = []
    num_replacements = max(1, int(len(words) * replacement_ratio))

    for word, pos in pos_tag(words):
        if not word.isalpha():
            altered_words.append(word)
            continue

        if num_replacements > 0 and random.random() < replacement_ratio:
            synonyms = get_synonyms(word, replace_synonyms, pos)
            hypernyms = get_hypernyms(word, replace_hypernyms, pos)
            negated_antonym = get_negated_antonym(word, replace_antonyms)

            alternatives = synonyms + hypernyms
            if replace_antonyms and negated_antonym:
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


def get_synonyms(word, replace_synonyms, pos):
    if not replace_synonyms:
        return []

    wordnet_pos = get_wordnet_pos(pos)
    synonyms = set()

    for syn in wordnet.synsets(word, pos=wordnet_pos):
        for lemma in syn.lemmas():
            if lemma.name() != word:  # Avoid the word itself
                synonyms.add(lemma.name().replace('_', ' '))

    return list(synonyms)


def get_hypernyms(word, replace_hypernyms, pos):
    if not replace_hypernyms:
        return []

    wordnet_pos = get_wordnet_pos(pos)
    hypernyms = set()

    for syn in wordnet.synsets(word, pos=wordnet_pos):
        for hypernym in syn.hypernyms():
            hypernyms.update([lemma.name().replace('_', ' ') for lemma in hypernym.lemmas()])

    return list(hypernyms)


def get_negated_antonym(word, replace_antonyms):
    if not replace_antonyms:
        return None
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # check if antonyms exist and return the negated antonym
            if lemma.antonyms():
                return f"not {lemma.antonyms()[0].name().replace('_', ' ')}"
    return None


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('NN'):
        return wordnet.NOUN
    elif treebank_tag.startswith('VB'):
        return wordnet.VERB
    elif treebank_tag.startswith('JJ'):
        return wordnet.ADJ
    elif treebank_tag.startswith('RB'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to NOUN if unknown


def stylometric_analysis(text):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    sentences = nltk.sent_tokenize(text)
    print(f"discovered sentences {sentences}")

    num_words = len(words)
    num_sentences = len(sentences)
    num_characters = sum(len(word) for word in words)
    avg_word_length = num_characters / num_words if num_words > 0 else 0
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

    word_frequencies = Counter(words)

    print("Stylometric Analysis:")
    print(f"Total words: {num_words}")
    print(f"Total characters (excluding spaces): {num_characters}")
    print(f"Average word length: {avg_word_length:.2f}")
    print(f"Average sentence length (in words): {avg_sentence_length:.2f}")

    print(f"Word frequencies: {word_frequencies.most_common()}")

    words_memo = word_frequencies.most_common(10)
    words, frequencies = zip(*words_memo)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(words, frequencies, color='lightblue', edgecolor='black')

    plt.title('Most common 10 word frequencies')
    plt.xlabel('Words')
    plt.ylabel('Frequencies')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, frequency in zip(bars, frequencies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(frequency),
                 ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()


def read_content(filename, read_from_console=False):
    if read_from_console:
        text = input('Enter some text here: ')
        print(text)

        return text

    try:
        with open(filename, "r") as f:
            content = f.read()
            print(content)

        return content
    except FileNotFoundError:
        print("File not found")


def keywords_extraction(text):
    preprocessed_text = preprocess_text(text)

    keywords_tfidf = extract_keywords_tfidf(preprocessed_text, n_keywords=3)
    # keywords_rake = extract_keywords_rake(preprocessed_text, n_keywords=3)

    print(keywords_tfidf)

    sentences_for_keywords_tfidf = generate_sentences_for_keywords(text, keywords_tfidf)
    # sentences_for_keywords_rake = generate_sentences_for_keywords(text, keywords_rake)

    print("Sentences for TF-IDF Keywords:")
    for sentence in sentences_for_keywords_tfidf:
        print(sentence)

    # print("\nSentences for RAKE Keywords:")
    # for sentence in sentences_for_keywords_rake:
    #    print(sentence)


if __name__ == '__main__':
    generator = pipeline('text-generation', model='HuggingFaceTB/SmolLM-135M', device=0)

    filename = 'input_file.txt'
    contents = read_content(filename, False)

    language = detect(contents)
    print(f"Detected language: {language}")

    probabilities = detect_langs(contents)
    print(f"Language probabilities: {probabilities}")

    stylometric_analysis(contents)
    # generate_alternative_text(contents, 0.2, True, True, True)
    # text = "The house is far from the city."
    """
    #alternative_text = generate_alternative_text(text, replacement_ratio=0.2, replace_synonyms=True,
    #                                             replace_hypernyms=True, replace_antonyms=True)
    """
    # print(alternative_text)

    content = "The house is far from the city. It is a beautiful structure located on a hill. The dog is happy playing in the yard."
    # with open(filename, "r") as f:
    #    content = f.read()
    content = "The house is far from the city. It is a beautiful structure located on a hill. The dog is happy playing in the yard."
    keywords_extraction(content)
    # generate_sentence_with_gpt("house", "The house is far from the city")
