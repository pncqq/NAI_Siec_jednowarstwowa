import functions


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
    max_err1 = 0.07
    weight_matrix, bias = functions.learn(vec1, weight_matrix, alpha, bias, max_err1)
    print("\n")

    # Testowanie
    bad_vectors = functions.test(vec2, weight_matrix, bias)
    functions.print_bad_vectors(bad_vectors)

    # Klasyfikacja z konsoli
    print("\n")
    one_text = functions.text_to_vector(input("Wpisz tekst w danym języku: "))
    functions.classify(one_text, weight_matrix, bias)


if __name__ == '__main__':
    main()
