import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Lab1_img import imgToFloat
import os
import time

pallet8 = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
])

pallet16 = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 0.5, 0.0],
    [0.5, 0.5, 0.5],
    [0.0, 1.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [1.0, 0.0, 0.0],
    [0.75, 0.75, 0.75],
    [0.0, 0.5, 0.5],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 0.0]
])


def generatePallet(n):
    if n == 8:
        return pallet8
    elif n == 16:
        return pallet16
    else:
        return np.linspace(0, 1, 2 ** n).reshape(2 ** n, 1)


def colorFit(pixel: float | np.ndarray[float], pallet: np.ndarray[float]) -> np.ndarray[float]:
    """
    Funkcja znajduję najbliższy kolor w tablicy koloru (pallet) do podanego piksela. Najpierw dla każdego piksela
    znajduje odległość do każdego koloru w tablicy koloru (pallet). Następnie zwraca kolor z palety, który ma
    najmniejszą odległość.
    Args:
        pixel: Kolor lub kolory, dla których chcemy znaleźć najbliższy kolor z palety.
        pallet: Paleta dostępnych kolorów.

    Returns:
        Kolor z palety, który ma najmniejszą odległość do podanego piksela.

    """
    distance = np.linalg.norm(pallet - pixel, axis=1)
    return pallet[np.argmin(distance)]


def quantization(picture: np.ndarray[float] | float, pallet: np.ndarray[float]) -> np.ndarray[float]:
    """
    Funkcja przeprowadza kwantyzację obrazu. Najpierw sprawdza, czy obraz jest typu float. Następnie dla każdego piksela
    obrazu wywołuje funkcję colorFit, która znajduje najbliższy kolor z palety do podanego piksela. Na koniec zwraca
    obraz.
    Args:
        picture: Macierz zawierająca obrazek
        pallet: Paleta dostępnych kolorów.

    Returns:
        Obraz po kwantyzacji.

    Raises:
        TypeError: Jeśli piksele nie są typu float.

    """
    if not np.issubdtype(picture.dtype, np.floating):
        raise TypeError("Typ danych w obrazie musi być float")

    rows, cols = picture.shape[:2]
    out_img = picture.copy()
    for w in tqdm(range(rows)):
        for k in range(cols):
            out_img[w, k] = colorFit(picture[w, k], pallet)
    return out_img


def dithering(picture: np.ndarray[float], method: str = 'random', pallet: np.ndarray[float] = pallet8):
    """
    Dithering to efekt zastosowania szumu, w celu zniwelowania błędu kwantyzacji. Funkcja przyjmuje obraz, metodę, oraz
    paletę kolorów. Sposób działania zależy od metody.

    Args:
        picture: Macierz zawierająca obrazek
        method: Metoda ditheringu. Możliwe wartości: 'random', 'ordered', 'floyd-steinberg'
        pallet: Paleta dostępnych kolorów.

    Returns:
        Obraz po ditheringu.

    Raises:
        TypeError: Jeśli piksele nie są typu float.

    """
    if not np.issubdtype(picture.dtype, np.floating):
        raise TypeError("Typ danych w obrazie musi być float")

    match method:
        case 'random':
            """
            Metoda ta polega na losowym wyborze piksela. Jeśli wartość piksela jest większa od losowej wartości z 
            przedziału [0, 1], to piksel przyjmuje wartość 1, w przeciwnym wypadku 0.
            Algorytm postępuję następująco:
            1. Obraz jest konwertowany do skali szarości (jeśli jest kolorowy).
            2. Pobierane są wymiary obrazu.
            3. Tworzona jest macierz o wymiarach obrazu, w której każdy piksel przyjmuje losowe wartość z przedziału 
            [0, 1].
            4. Obraz jest porównywany z macierzą losowych wartości. Jeśli wartość piksela jest większa od losowej 
            wartości, to piksel przyjmuje wartość 1, w przeciwnym wypadku 0.
            5. Zwracany jest obraz.
            """
            if picture.ndim == 3:
                picture = picture.mean(axis=2)
            rows, cols = picture.shape[:2]
            random = np.random.rand(rows, cols)
            out_img = (picture >= random) * 1
            return out_img
        case 'ordered':
            """
            Dithering zorganizowany to metoda, w której używa się kwantyzacji (funkcji colorFit) oraz mapy progowania.
            Algorytm postępuję następująco:
            1. Pobierane są wymiary obrazu.
            2. Kopiowany jest obraz.
            3. Wyświetlana jest lista dostępnych map progowania. Użytkownik wybiera mapę.
            4. Mapa progowania jest przekształcona do wartości przesunięcia (będzie zawierała wartości z przedziału
            [-0.5, 0.5]).
            5. Obliczany jest współczynnik skalowania koloru (r).
            6. Dla każdego piksela obrazu:
            6.1. Obliczana jest nowa wartość piksela (nowy kolor).
            7. Zwracany jest obraz.
            """
            rows, cols = picture.shape[:2]
            out_img = picture.copy()
            choice = input("Dithering zorganizowany. Wybierz mapę progowania:\n"
                           "1. 2x2\n"
                           "2. 4x4\n"
                           "Wybór: ")
            os.system('cls')
            match choice:
                case '1':
                    threshold = np.array([[0, 2],
                                          [3, 1]]) / 4
                case '2':
                    threshold = np.array([[0, 8, 2, 10],
                                          [12, 4, 14, 6],
                                          [3, 11, 1, 9],
                                          [15, 7, 13, 5]]) / 16
                case _:
                    print("Nie ma takiej opcji")
                    exit(1)
            mpre = (threshold + 1) / (int(choice) * 2) ** 2 - 0.5
            r = len(pallet) / len(picture)
            for w in tqdm(range(rows)):
                for k in range(cols):
                    out_img[w, k] = colorFit(picture[w, k] + r * mpre[w % int(choice), k % int(choice)], pallet)
            return out_img

        case 'floyd-steinberg':
            """
            Metoda Floyd-Steinberg to metoda ditheringu, która polega na przekazywaniu błędu kwantyzacji do 
            sąsiadujących pikseli (do piksela po prawej stronie, piksela pod oraz piksela na dole po prawej i lewej 
            stronie. 
            Algorytm postępuję następująco:
            1. Pobierane są wymiary obrazu.
            2. Kopiowany jest obraz.
            3. Dla każdego piksela obrazu:
            3.1. Obliczana jest nowa wartość piksela za pomocą kwantyzacji (nowy kolor).
            3.2. Obliczany jest błąd kwantyzacji (jest to różnica w wartości starego piksela z nowym).
            3.3 Jeśli piksel nie jest ostatnim pikselem w wierszu, to do piksela po prawej stronie dodawany jest błąd 
            pomnożony przez 7/16.
            3.4 Jeśli piksel nie jest ostatnim pikselem w kolumnie, to do piksela pod dodawany jest błąd pomnożony 
            przez 5/16.
            3.5 Jeśli piksel nie jest ostatnim pikselem w wierszu i nie jest pierwszym pikselem w kolumnie, to do
            piksela na dole po lewej stronie dodawany jest błąd pomnożony przez 3/16.
            3.6 Jeśli piksel nie jest ostatnim pikselem w wierszu i nie jest ostatnim pikselem w kolumnie, to do 
            piksela na dole po prawej stronie dodawany jest błąd pomnożony przez 1/16.
            4. Zwracany jest obraz.
            """
            rows, cols = picture.shape[:2]
            out_img = picture.copy()
            for w in tqdm(range(rows)):
                for k in range(cols):
                    out_img[w, k] = colorFit(picture[w, k], pallet)
                    error = picture[w, k] - out_img[w, k]
                    if k + 1 < cols:
                        picture[w, k + 1] += error * 7 / 16
                    if w + 1 < rows:
                        if k - 1 >= 0:
                            picture[w + 1, k - 1] += error * 3 / 16
                        picture[w + 1, k] += error * 5 / 16
                        if k + 1 < cols:
                            picture[w + 1, k + 1] += error * 1 / 16
            return out_img
    pass


def test_for_black_white_pictures(picture, file_name):
    picture = imgToFloat(picture)
    one_bit_pallet = generatePallet(1)
    new_picture = quantization(picture, one_bit_pallet)
    fig, axs = plt.subplot_mosaic([['Original', '1 bit'], ['2 bit', '4 bit']], figsize=(10, 10))
    fig.suptitle(f"{file_name} - Czysta kwantyzacja")
    axs['Original'].imshow(picture, cmap="gray")
    axs['Original'].set_title("Original")
    axs['1 bit'].imshow(new_picture, cmap="gray")
    axs['1 bit'].set_title("1 bit")
    axs['2 bit'].imshow(quantization(picture, pallet=generatePallet(2)), cmap="gray")
    axs['2 bit'].set_title("2 bit")
    axs['4 bit'].imshow(quantization(picture, pallet=generatePallet(4)), cmap="gray")
    axs['4 bit'].set_title("4 bit")
    plt.show()

    fig1, axs1 = plt.subplot_mosaic([['Original', '1 bit'], ['2 bit', '4 bit']], figsize=(10, 10))
    fig1.suptitle(f"{file_name} - Dithering losowy")
    axs1['Original'].imshow(picture, cmap="gray")
    axs1['Original'].set_title("Original")
    axs1['1 bit'].imshow(dithering(picture, pallet=generatePallet(1)), cmap="gray")
    axs1['1 bit'].set_title("1 bit")
    axs1['2 bit'].imshow(dithering(picture, pallet=generatePallet(2)), cmap="gray")
    axs1['2 bit'].set_title("2 bit")
    axs1['4 bit'].imshow(dithering(picture, pallet=generatePallet(4)), cmap="gray")
    axs1['4 bit'].set_title("4 bit")
    plt.show()

    fig2, axs2 = plt.subplot_mosaic([['Original', '1 bit'], ['2 bit', '4 bit']], figsize=(10, 10))
    fig2.suptitle(f"{file_name} - Dithering zorganizowany")
    axs2['Original'].imshow(picture, cmap="gray")
    axs2['Original'].set_title("Original")
    axs2['1 bit'].imshow(dithering(picture, method="ordered", pallet=generatePallet(1)), cmap="gray")
    axs2['1 bit'].set_title("1 bit")
    axs2['2 bit'].imshow(dithering(picture, method="ordered", pallet=generatePallet(2)), cmap="gray")
    axs2['2 bit'].set_title("2 bit")
    axs2['4 bit'].imshow(dithering(picture, method="ordered", pallet=generatePallet(4)), cmap="gray")
    axs2['4 bit'].set_title("4 bit")
    plt.show()

    fig3, axs3 = plt.subplot_mosaic([['Original', '1 bit'], ['2 bit', '4 bit']], figsize=(10, 10))
    fig3.suptitle(f"{file_name} - Dithering metodą Floyd-Steinberg'a")
    axs3['Original'].imshow(picture, cmap="gray")
    axs3['Original'].set_title("Original")
    axs3['1 bit'].imshow(dithering(picture, method="floyd-steinberg", pallet=generatePallet(1)), cmap="gray")
    axs3['1 bit'].set_title("1 bit")
    axs3['2 bit'].imshow(dithering(picture, method="floyd-steinberg", pallet=generatePallet(2)), cmap="gray")
    axs3['2 bit'].set_title("2 bit")
    axs3['4 bit'].imshow(dithering(picture, method="floyd-steinberg", pallet=generatePallet(4)), cmap="gray")
    axs3['4 bit'].set_title("4 bit")
    plt.show()


def test_for_color_pictures(picture, file_name):
    picture = imgToFloat(picture)
    fig, axs = plt.subplot_mosaic([['Original', '8 bit', '16 bit']], figsize=(10, 10))
    fig.suptitle(f"{file_name} - Czysta kwantyzacja")
    axs['Original'].imshow(picture)
    axs['Original'].set_title("Original")
    axs['8 bit'].imshow(quantization(picture, pallet=generatePallet(8)))
    axs['8 bit'].set_title("8 bit")
    axs['16 bit'].imshow(quantization(picture, pallet=generatePallet(16)))
    axs['16 bit'].set_title("16 bit")
    plt.show()

    fig1, axs1 = plt.subplot_mosaic([['Original', '8 bit', '16 bit']], figsize=(10, 10))
    fig1.suptitle(f"{file_name} - Dithering losowy")
    axs1['Original'].imshow(picture)
    axs1['Original'].set_title("Original")
    axs1['8 bit'].imshow(dithering(picture, pallet=generatePallet(8)), cmap="gray")
    axs1['8 bit'].set_title("8 bit")
    axs1['16 bit'].imshow(dithering(picture, pallet=generatePallet(16)), cmap="gray")
    axs1['16 bit'].set_title("16 bit")
    plt.show()

    fig2, axs2 = plt.subplot_mosaic([['Original', '8 bit', '16 bit']], figsize=(10, 10))
    fig2.suptitle(f"{file_name} - Dithering zorganizowany")
    axs2['Original'].imshow(picture)
    axs2['Original'].set_title("Original")
    axs2['8 bit'].imshow(dithering(picture, method="ordered", pallet=generatePallet(8)))
    axs2['8 bit'].set_title("8 bit")
    axs2['16 bit'].imshow(dithering(picture, method="ordered", pallet=generatePallet(16)))
    axs2['16 bit'].set_title("16 bit")
    plt.show()

    fig3, axs3 = plt.subplot_mosaic([['Original', '8 bit', '16 bit']], figsize=(10, 10))
    fig3.suptitle(f"{file_name} - Dithering metodą Floyd-Steinberg'a")
    axs3['Original'].imshow(picture)
    axs3['Original'].set_title("Original")
    axs3['8 bit'].imshow(dithering(picture, method="floyd-steinberg", pallet=generatePallet(8)))
    axs3['8 bit'].set_title("8 bit")
    axs3['16 bit'].imshow(dithering(picture, method="floyd-steinberg", pallet=generatePallet(16)))
    axs3['16 bit'].set_title("16 bit")
    plt.show()


if __name__ == '__main__':
    print("Co chcesz zrobić?\n"
          "1. Przetestować dithering losowy\n"
          "2. Przetestować dithering zorganizowany\n"
          "3. Przetestować dithering metodą Floyd'a-Steinberg'a\n"
          "4. Przeprowadzić pełne badanie\n"
          "5. Przetestować czystą kwantyzację\n")
    choice = int(input("Podaj liczbę: "))
    os.system("cls")

    match choice:
        case 1:
            for file in os.listdir("SM_Lab04"):
                print(file)
            file = input("Wybierz plik: ")
            os.system("cls")
            img = None
            try:
                print("Wczytuje obraz...")
                img = plt.imread(f'SM_Lab04/{file}')
                print("Sukces!")
                time.sleep(2)
            except FileNotFoundError as e:
                print(e)
                exit(1)
            img = imgToFloat(img)
            n = int(input("Na ilu bitach zapisać: "))
            new_img = dithering(img, pallet=generatePallet(n))
            plt.imshow(new_img, cmap="gray")
            plt.title(f"Dithering losowy - {n} bity")
            plt.show()

        case 2:
            for file in os.listdir("SM_Lab04"):
                print(file)
            file = input("Wybierz plik: ")
            os.system("cls")
            img = None
            try:
                print("Wczytuje obraz...")
                img = plt.imread(f'SM_Lab04/{file}')
                print("Sukces!")
                time.sleep(2)
            except FileNotFoundError as e:
                print(e)
                exit(1)
            img = imgToFloat(img)
            n = int(input("Na ilu bitach zapisać: "))
            new_image = dithering(img, method="ordered", pallet=generatePallet(n))
            plt.imshow(new_image)
            plt.title(f"Dithering zorganizowany - {n} bity")
            plt.show()

        case 3:
            for file in os.listdir("SM_Lab04"):
                print(file)
            file = input("Wybierz plik: ")
            os.system("cls")
            img = None
            try:
                print("Wczytuje obraz...")
                img = plt.imread(f'SM_Lab04/{file}')
                print("Sukces!")
                time.sleep(2)
            except FileNotFoundError as e:
                print(e)
                exit(1)
            img = imgToFloat(img)
            n = int(input("Na ilu bitach zapisać: "))
            new_image = dithering(img, method="floyd-steinberg", pallet=generatePallet(n))
            plt.imshow(new_image)
            plt.title(f"Dithering metodą Floyd-Steinberg'a - {n} bity")
            plt.show()

        case 4:
            picture_matrix = None

            files = {}
            while True:
                os.system("cls")
                for file in os.listdir("SM_Lab04"):
                    print(file)
                file = input("Wybierz pliki (wpisz 0 by zakończyć dodawanie): ")
                if file == "0":
                    break
                type = input("Wybierz typ: \n"
                             "1. Kolorowy\n"
                             "2. Czarno-biały\n")
                files[file] = type
            try:
                for file in files:
                    os.system("cls")
                    picture_matrix = plt.imread(f'SM_Lab04/{file}')
                    print(f"Otworzono plik {file}!")
                    time.sleep(1)
                    os.system("cls")
                    if files[file] == "1":
                        test_for_color_pictures(picture_matrix, file)
                    elif files[file] == "2":
                        test_for_black_white_pictures(picture_matrix, file)
            except FileNotFoundError as e:
                print(e)

        case 5:
            for file in os.listdir("SM_Lab04"):
                print(file)
            file = input("Wybierz plik: ")
            os.system("cls")
            img = None
            try:
                print("Wczytuje obraz...")
                img = plt.imread(f'SM_Lab04/{file}')
                print("Sukces!")
                time.sleep(1)
            except FileNotFoundError as e:
                print(e)
                exit(1)
            img = imgToFloat(img)
            n = int(input("Na ilu bitach zapisać: "))
            new_image = quantization(img, pallet=generatePallet(n))
            plt.imshow(new_image,cmap="gray")
            plt.title(f"Kwantyzacja - {n} bity")
            plt.show()

        case _:
            print("Nie prawidłowy wybór")
            exit(0)
