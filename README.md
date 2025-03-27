# 🧬 NAI_Siec_jednowarstwowa

Projekt przedstawia implementację **sieci jednowarstwowej (perceptronu)** do klasyfikacji języka tekstu na podstawie częstości występowania liter w zdaniu.

## 📖 Cel projektu

Zbudowanie prostej sieci neuronowej rozpoznającej język tekstu spośród czterech możliwych: **angielski, niemiecki, polski, hiszpański**, na podstawie analizy częstotliwości liter alfabetu łacińskiego (26 liter, bez znaków diakrytycznych).

## 📂 Zawartość repozytorium

- `main.py` – główna logika aplikacji
- `functions.py` – pomocnicze funkcje (np. preprocessing, predykcja)
- `Data/lang.train.csv` – dane treningowe
- `Data/lang.test.csv` – dane testowe
- `.idea/` – pliki konfiguracyjne PyCharm

## ⚙️ Technologie

- Python 3.x
- NumPy

## 🚀 Jak to działa?

1. Teksty zamieniane są na wektory 26 liczb (liczba wystąpień liter a-z).
2. Wektor jest normalizowany:
   \[ \hat{v} = \frac{v}{|v|} \]
3. Uczenie perceptronu następuje przy użyciu funkcji liniowej lub progowej.
4. Do każdego języka przypisany jest osobny neuron.
5. W fazie testu wybierany jest neuron o najwyższej aktywacji.

## 🔄 Funkcjonalności

- Trening sieci na zbiorze `lang.train.csv`
- Testowanie skuteczności na `lang.test.csv`
- Obsługa wklejenia tekstu z klawiatury i rozpoznania języka
- (Opcjonalnie) wypisanie błędnie sklasyfikowanych tekstów

## ▶️ Jak uruchomić

1. Sklonuj repo:
```bash
git clone https://github.com/pncqq/NAI_Siec_jednowarstwowa.git
cd NAI_Siec_jednowarstwowa
```

2. Uruchom program:
```bash
python main.py
```

> 🔎 Program automatycznie wczyta dane z katalogu `Data/`. W konsoli można wkleić nowy tekst do klasyfikacji.

## 👨‍💻 Autor
**Filip Michalski**  
Projekt wykonany w ramach kursu NAI (Narzędzia AI) jako praktyczne wprowadzenie do sieci neuronowych i klasyfikacji tekstu.
