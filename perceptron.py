import numpy as np
import functions as f


def learning_process(training_set, th, alfa, weights_vector, test_set):
    # 0. Dzialaj, dopoki bład iteracji nie bedzie mniejszy
    it_err = 2

    # while it_err > 0.015:
    for i in range(1000):
        # Zmienne
        y_values = []
        d_values = []
        i = 0

        for vec_x in training_set:
            # 1. Obliczamy wartość y (klasyfikacja gatunków Iris)
            # Obliczamy net
            dot_product = np.dot(weights_vector, vec_x[:-1])  # iloczyn skalarny
            net = dot_product - th

            # Obliczamy wartość wyjściową y
            if net > th:
                y_values.append(1)
            else:
                y_values.append(0)

            # 2. Aktualizujemy wagi i odchylenie (metoda delta)
            # Dodajemy oczekiwaną wartość wyjściową (argument decyzyjny wektora)
            d = vec_x[-1]
            d_values.append(d)

            # Obliczamy nowy wektor wag
            multiplier = alfa * (d - y_values[i])
            input_vec_updated = multiplier * np.array(vec_x[:-1])
            weights_vector = np.add(weights_vector, input_vec_updated)

            # Obliczamy nowe odchylenie
            th = th - multiplier

            # Iterujemy indeks
            i = i + 1

        # 3. Wyznaczanie błędu iteracji
        sum = 0
        for i in range(1, len(training_set)):
            sum = sum + (d_values[i] - y_values[i]) ** 2

        it_err = 1 / len(training_set) * sum
        print(f"Błąd iteracji wynosi {it_err}")

    print("Perceptron nauczony!")
    print("Rozpoczynam testy dla danych testowych...")

    # Testujemy dla danych testowych ------------------------------------------------------------------------------------
    # Zmienne
    y_values = []
    d_values = []
    i = 0

    for vec_x in test_set:
        # 1. Obliczamy wartość y (klasyfikacja gatunków Iris)
        # Obliczamy net
        dot_product = np.dot(weights_vector, vec_x[:-1])  # iloczyn skalarny
        net = dot_product - th

        # Obliczamy wartość wyjściową y
        if net > th:
            y_values.append(1)
        else:
            y_values.append(0)

        # # 2. Aktualizujemy wagi i odchylenie (metoda delta)
        # # Dodajemy oczekiwaną wartość wyjściową (argument decyzyjny wektora)
        d = vec_x[-1]
        d_values.append(d)
        #
        # # Obliczamy nowy wektor wag
        # multiplier = alfa * (d - y_values[i])
        # input_vec_updated = multiplier * np.array(vec_x[:-1])
        # weights_vector = np.add(weights_vector, input_vec_updated)
        #
        # # Obliczamy nowe odchylenie
        # th = th - multiplier

        # Iterujemy indeks
        i = i + 1

    # 3. Wyznaczanie dokładności
    i = 0
    acc = 0
    for vec in test_set:
        if vec[-1] == y_values[i]:
            acc = acc + 1
        i = i + 1
    accuracy = acc / len(y_values)

    print(f"Dokładność przewidywań na zbiorze testowym wynosi: {accuracy * 100} %!")


def main():
    # Wektory wejściowe wraz z oczekiwanymi wartościami wyjściowymi
    training_set = f.load_data("Data/perceptron.data")
    test_set = f.load_data("Data/perceptron.test.data")

    # Inicjalizacja randomowych wag
    weights_vector = f.random_weight_vector()

    # Wybór stałej uczenia alfa i odchylenia theta
    f.alfa = float(input("Podaj stałą uczenia alfa: "))
    theta = float(input("Podaj odchylenie theta: "))

    # Uczymy perceptron oraz zapisujemy błąd iteracji
    learning_process(training_set, theta, f.alfa, weights_vector, test_set)


