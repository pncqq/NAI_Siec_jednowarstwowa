import collections

import pandas as pd
import random as rnd


# Wczytywanie danych z pliku CSV
def load_data(file_name):
    data = pd.read_csv(file_name, usecols=[0, 1], header=None)

    text = data[1].tolist()
    labels = data[0].tolist()

    # Zliczanie ile jest języków
    languages = set()
    for i in labels:
        languages.add(i)

    lang_count = len(languages)

    return text, labels, lang_count


def random_weight_vector(dimension_count):
    weights_vector = []
    # Wagi początkowe wektora wag
    for i in range(dimension_count):
        val = rnd.random()
        weights_vector.append(val)

    return weights_vector


def text_to_vector(text_list):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    # Tworzymy listę wektorów dla każdego tekstu
    vectors = []
    for text in text_list:
        # Tworzymy słownik liczników, jeden dla każdej litery alfabetu
        counts = collections.defaultdict(int)
        for char in text.lower():
            # Jeśli litera jest w alfabecie, zwiększamy jej licznik o 1
            if char in alphabet:
                counts[char] += 1
        # Zamieniamy słownik na wektor i normalizujemy go
        vector = [counts[letter] for letter in alphabet]
        total = sum(vector)
        normalized_vector = [count / total for count in vector]
        vectors.append(normalized_vector)
    return vectors


def activation_function(net):
    # TODO napisac liniowa funkcje akt
    return 1


def map_labels_to_integers(labels):
    label_map = {}
    integer_labels = []
    integer_label = 0
    for label in labels:
        if label not in label_map:
            label_map[label] = integer_label
            integer_label += 1
        integer_labels.append(label_map[label])
    return integer_labels
