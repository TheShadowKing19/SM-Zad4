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
    if picture.dtype != float:
        raise TypeError("Typ danych w obrazie musi być float")

    match method:
        case 'random':
            if picture.ndim == 3:
                picture = picture.mean(axis=2)
            rows, cols = picture.shape[:2]
            random = np.random.rand(rows, cols)
            out_img = (picture >= random) * 1
            return out_img
        case 'ordered':
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
            raise NotImplementedError
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

    changed_img = dithering(img, 'ordered', pallet16)
    plt.imshow(changed_img)
    plt.show()



