import collections
import csv
import numpy as np

import random as rnd

# Zmienne globalne

label_to_numeric = {}
dimension_count = 0
letters_count = 26


# Wczytywanie danych z pliku CSV
def load_data(file_name):
    # Wczytywanie pliku do listy
    data = []
    with open(file_name, 'r', encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            label = row[0]
            if label not in label_to_numeric:
                label_to_numeric[label] = len(label_to_numeric)

            # Etykieta danego jezyka
            numeric_label = label_to_numeric[label]

            row_data = []
            for idx, text in enumerate(row):
                if idx != len(row) - 1:
                    continue
                else:
                    row_data.append(text)
                    row_data.append(numeric_label)
            data.append(row_data)

    # Ile języków (czyli ile klas)
    global dimension_count
    dimension_count = len(label_to_numeric)
    return data


# Generuje macierz wag
def random_weight_matrix():
    return np.random.rand(dimension_count, letters_count)


def random_bias_matrix():
    return np.random.rand(dimension_count, 1)


# Generuje wekto znormalizowany
def text_to_vector(text_with_labels):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    # Tworzymy listę wektorów dla każdego tekstu
    vectors = []

    for i, text_list in enumerate(text_with_labels):
        for text in text_list:
            # Sprawdzamy, czy mamy label czy tekst
            if isinstance(text, int):
                continue

            # Tworzymy słownik liczników, jeden dla każdej litery alfabetu
            counts = collections.defaultdict(int)
            for char in text.lower():
                # Jeśli litera jest w alfabecie, zwiększamy jej licznik o 1
                if char in alphabet:
                    counts[char] += 1

            # Zamieniamy słownik na wektor i normalizujemy go
            vector = [counts[letter] for letter in alphabet]
            total = sum(vector)

            # Normalizujemy wektor (czestotliwosc wystapien danej liczby w kolejnosci alfabetycznej)
            normalized_vector = [count / total for count in vector]

            # Dodajemy na koncu etykiete języka
            normalized_vector.append(text_with_labels[i][1])

            vectors.append(normalized_vector)
    return vectors


# Liniowa funkcja aktywacji
def activation_function(net):
    return net


def learn(input_vectors, weight_matrix, alpha, bias):
    # Przechodzimy przez wektory wejsciowe
    for vec in input_vectors:
        # Obliczamy W * x
        x = np.array(vec[:-1])
        Wx = np.matmul(weight_matrix, x)
        # Odejmujemy bias (funkcja liniowa, wiec net = f(net))
        biasT = np.transpose(bias)
        net = Wx - biasT

        # Rozszerzamy oczekiwaną wartość wyjściową na macierz
        d = vec[-1]
        d_matrix = np.zeros(4)
        if d == 0:  # GR
            d_matrix[0] = 1
        elif d == 1:  # PL
            d_matrix[1] = 1
        elif d == 2:  # EN
            d_matrix[2] = 1
        else:  # SP
            d_matrix[3] = 1

        # Obliczamy miarę błędu
        diff = d_matrix - net
        E = 0.5 * np.dot(diff, np.transpose(diff))
        E = E[0][0]

        # Aktualizacja wag metodą gradientową
        # f'(net) = 1 -> bo liniowa
        diff_matrix = alpha * diff * 1
        diff_matrix = diff_matrix[0].tolist()
        diff_matrix = np.array(diff_matrix)
        diff_matrix = diff_matrix.reshape((4, 1))

        x = x.tolist()
        x = np.array(x)
        x = x.reshape((1, 26))

        # Mnożenie macierzy
        matrix = np.dot(diff_matrix, x)

        new_weights = weight_matrix + matrix

        # Aktualizacja bias
        new_bias = bias - diff_matrix
