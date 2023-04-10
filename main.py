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
    # Wczytywanie plikow
    train_text, train_labels, lang_count = functions.load_data("Data/lang.train.csv")
    test_text, test_labels, lang_count1 = functions.load_data("Data/lang.test.csv")

    # Zakonczenie gdy pliki sie nie zgadzaja ze soba
    if lang_count != lang_count1:
        quit(997)

    # Treningowe
    input_vectors = functions.text_to_vector(train_text)
    weights_vector = functions.random_weight_vector(lang_count)

    # Mapowanie języków
    mapped_labels = functions.map_labels_to_integers(train_labels)

    # Trenowanie
    train_perceptron(input_vectors, mapped_labels, 0.5, 50)


if __name__ == '__main__':
    main()
