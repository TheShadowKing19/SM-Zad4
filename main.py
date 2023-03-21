import numpy as np
import matplotlib.pyplot as plt

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


def quantization(picture, pallet):
    rows, cols = picture.shape[:2]
    out_img = picture.copy()
    for w in range(rows):
        for k in range(cols):
            out_img[w, k] = colorFit(picture[w, k], pallet)
    return out_img


if __name__ == '__main__':
    N = 3
    paleta = np.linspace(0, 1, 3).reshape(N, 1).astype(np.float32)
    img = None
    try:
        img = plt.imread('SM_Lab04/0001.jpg')
    except FileNotFoundError as e:
        print(e)
    plt.imshow(quantization(img, pallet8))
    plt.show()


