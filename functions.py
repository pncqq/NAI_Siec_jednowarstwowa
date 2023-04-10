import collections
import csv
import numpy as np

import random as rnd

# Zmienne globalne

label_to_numeric = {}
dimension_count = 0
letters_count = 26
all_normalized_vectors = []


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


# Generuje bias
def random_bias_matrix():
    return np.random.rand(dimension_count, 1)


# Generuje wektor znormalizowany
def text_to_vector(text_with_labels):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    # Jeśli to tylko jeden tekst
    if isinstance(text_with_labels, str):
        # Tworzymy słownik liczników, jeden dla każdej litery alfabetu
        counts = collections.defaultdict(int)
        for char in text_with_labels.lower():
            # Jeśli litera jest w alfabecie, zwiększamy jej licznik o 1
            if char in alphabet:
                counts[char] += 1

        # Zamieniamy słownik na wektor i normalizujemy go
        vector = [counts[letter] for letter in alphabet]
        total = sum(vector)

        # Normalizujemy wektor (czestotliwosc wystapien danej liczby w kolejnosci alfabetycznej)
        normalized_vector = [count / total for count in vector]
        return normalized_vector

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

    global all_normalized_vectors
    all_normalized_vectors = vectors
    return vectors


# Liniowa funkcja aktywacji
def activation_function(net):
    return net


# Uczenie
def learn(input_vectors, weight_matrix, alpha, bias, max_err):
    E = 1

    while E > max_err:
        # Przechodzimy przez wektory wejsciowe
        for vec in input_vectors:
            # Obliczamy W * x
            x = np.array(vec[:-1])
            Wx = np.matmul(weight_matrix, x)
            # Odejmujemy bias (funkcja liniowa, wiec net = f(net))
            biasT = np.transpose(bias)
            # Klasyfikacja
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
            weight_matrix = new_weights

            # Aktualizacja bias
            new_bias = bias - diff_matrix
            bias = new_bias

        print("Sieć nauczona!")
        print(f"Miara błędu wynosi: {E}")
    return weight_matrix, bias


# Testowanie
def test(input_vectors, weight_matrix, bias):
    good = 0
    all = len(input_vectors)
    bad_vectors = []

    # Przechodzimy przez wektory wejsciowe
    for vec in input_vectors:
        # Obliczamy W * x
        x = np.array(vec[:-1])
        Wx = np.matmul(weight_matrix, x)
        # Odejmujemy bias (funkcja liniowa, wiec net = f(net)) - WARTOŚĆ NET
        biasT = np.transpose(bias)
        # Klasyfikacja
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

        # Sprawdzanie klasyfikacji
        classified = np.argmax(net)
        real_lang = np.argmax(d_matrix)
        if classified == real_lang:
            good = good + 1
        else:
            bad_vectors.append(vec)

    print("Sieć przetestowana!")
    acc = good / all * 100
    print(f"Dokładność wynosi: {acc}%")

    return bad_vectors


# Klasyfikacja jednego tekstu
def classify(vec, weight_matrix, bias):
    # Obliczamy W * x
    x = np.array(vec)
    Wx = np.matmul(weight_matrix, x)
    # Odejmujemy bias (funkcja liniowa, wiec net = f(net)) - WARTOŚĆ NET
    biasT = np.transpose(bias)
    # Klasyfikacja
    net = Wx - biasT

    # Sprawdzanie klasyfikacji
    print("Wykryto język:")
    classified = np.argmax(net)
    if classified == 0:
        print("Niemiecki")
    elif classified == 1:
        print("Polski")
    elif classified == 2:
        print("Angielski")
    elif classified == 3:
        print("Hiszpański")


def print_bad_vectors(bad_vectors):
    # TODO: wypisać te teksty ze zbioru testowego, dla których klasyfikacja była błędna.
    pass
