import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Lab1_img import imgToFloat


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
    if picture.dtype != float:
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
    if picture.dtype != float:
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
            choice = input("Wybierz mapę progowania:\n"
                           "1. 2x2\n"
                           "2. 4x4\n"
                           "Wybór: ")

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


if __name__ == '__main__':
    N = 3
    paleta = np.linspace(0, 1, 3).reshape(N, 1).astype(np.float32)
    img = None
    try:
        img = plt.imread('SM_Lab04/0011.jpg')
    except FileNotFoundError as e:
        print(e)
        exit(1)
    img = imgToFloat(img)
    # changed_img = quantization(img, pallet8)
    # plt.imshow(changed_img)
    # plt.show()

    # changed_img = dithering(img, 'random')
    # plt.imshow(changed_img, cmap='gray')
    # plt.show()

    # changed_img = dithering(img, 'ordered', pallet16)
    # plt.imshow(changed_img)
    # plt.show()
    changed_img = dithering(img, 'floyd-steinberg', pallet16)
    plt.imshow(changed_img)
    plt.show()



