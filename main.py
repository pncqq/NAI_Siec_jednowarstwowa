import numpy as np

import functions


def train_perceptron(inputs, labels, alpha, epochs, bias=np.zeros(4), weights=np.zeros((26, 4))):
    # pętla treningowa
    for epoch in range(epochs):
        # inicjujemy sumaryczny błąd na 0
        total_error = 0.0

        # przechodzimy po każdym wektorze wejściowym i etykiecie
        for input_vector, label in zip(inputs, labels):
            # obliczamy sumę iloczynów wag i wejść oraz dodajemy odchylenie
            net = np.dot(input_vector, weights) + bias

            # obliczamy wyjście neuronu
            output = functions.activation_function(net)

            # obliczamy błąd dla aktualnego wektora wejściowego i etykiety
            error = label - output

            # obliczamy zmianę wag na podstawie reguły delta
            delta_weights = np.outer(input_vector, alpha * error)
            delta_bias = alpha * error

            # aktualizujemy wagi i bias
            weights += delta_weights[:, np.newaxis]
            bias += delta_bias

            # dodajemy błąd do sumarycznego błędu
            total_error += np.sum(np.abs(error))

        # drukujemy informację o błędzie dla każdej epoki
        print(f"Epoka {epoch + 1}: błąd = {total_error:.3f}")

    return weights, bias


def main():
    # Wczytywanie plikow w formacie TEKST-VALUE
    train_text = functions.load_data("Data/lang.train.csv")
    test_text = functions.load_data("Data/lang.test.csv")

    # Przerabianie ww. na znormalizowane wektory
    vec1 = functions.text_to_vector(train_text)
    vec2 = functions.text_to_vector(test_text)

    # Generowanie randomowej macierzy wag
    weight_matrix = functions.random_weight_matrix()

    # Generowanie odchylenia
    bias = functions.random_bias_matrix()

    # Stała uczenia
    alpha = 1

    # Uczenie
    functions.learn(vec1, weight_matrix, alpha, bias)


if __name__ == '__main__':
    main()
