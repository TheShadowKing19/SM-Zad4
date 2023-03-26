import numpy as np
import matplotlib.pyplot as plt
import cv2


def imgToUInt8(img):
    if np.issubdtype(img.dtype, np.integer):
        pass

    img = img.astype('uint8') * 255

    return img


def imgToFloat(img):
    if np.issubdtype(img.dtype, np.floating):
        pass

    img = img / 255.0

    return img


if __name__ == '__main__':
    img1 = plt.imread('IMG_INTRO\A1.png')
    print(f"Przed konwersją:\n {img1} \n Typ danych: {img1.dtype}")
    img1 = imgToUInt8(img1)
    print(f"Po konwersji: \n {img1} \n Typ danych: {img1.dtype}")

    img2 = plt.imread('IMG_INTRO\A2.jpg')
    print(f"Przed konwersją:\n {img2} \n Typ danych: {img2.dtype}")
    img2 = imgToFloat(img2)
    print(f"Po konwersji: \n {img2} \n Typ danych: {img2.dtype}")

    img3 = plt.imread('IMG_INTRO\B01.png')
    fig, axs = plt.subplot_mosaic([['original', 'y1', 'y2'],
                                   ['r', 'g', 'b'],
                                   ['R', 'G', 'B']])

    axs['original'].imshow(img3)
    axs['original'].set_title("Oryginalny")

    y1 = 0.299 * img3[:, :, 0] + 0.587 * img3[:, :, 1] + 0.144 * img3[:, :, 2]
    axs['y1'].imshow(y1, cmap='gray')
    axs['y1'].set_title("y1")

    y2 = 0.2126 * img3[:, :, 0] + 0.7152 * img3[:, :, 1] + 0.0722 * img3[:, :, 2]
    axs['y2'].imshow(y2, cmap='gray')
    axs['y2'].set_title("y2")

    r = img3[:, :, 0]
    axs['r'].imshow(r, cmap='gray')
    axs['r'].set_title("r")

    g = img3[:, :, 1]
    axs['g'].imshow(g, cmap='gray')
    axs['g'].set_title("g")

    b = img3[:, :, 2]
    axs['b'].imshow(b, cmap='gray')
    axs['b'].set_title("b")

    # img3[:, :, 1] = 0
    R = img3[:, :, 0] + 0 * img3[:, :, 1] + 0 * img3[:, :, 2]
    axs['R'].imshow(R)
    axs['R'].set_title("R")

    G = 0 * img3[:, :, 0] + img3[:, :, 1] + 0 * img3[:, :, 2]
    axs['G'].imshow(G, cmap="Greens")
    axs['G'].set_title("G")

    B = 0 * img3[:, :, 0] + 0 * img3[:, :, 1] + img3[:, :, 2]
    axs['B'].imshow(B, cmap="Blues")
    axs['B'].set_title("B")
    plt.show()

    fig2, ax2 = plt.subplot_mosaic([['before', 'after']])
    img1_cv = cv2.imread('IMG_INTRO\A4.jpg')
    img_BGR = cv2.cvtColor(img1_cv, cv2.COLOR_RGB2BGR)
    ax2['before'].imshow(img1_cv)
    ax2['before'].set_title("cv2.imread")
    ax2['after'].imshow(img_BGR)
    ax2['after'].set_title("Po zmianie na BGR")
    plt.show()

    img4 = plt.imread("IMG_INTRO/B02.jpg")
    fragment = img4[600:800, 600:800]
    fig3, ax3 = plt.subplot_mosaic([['Tecza', 'wycinek']])
    ax3['Tecza'].imshow(img4)
    ax3['Tecza'].set_title("Oryginalna")
    ax3['wycinek'].imshow(fragment)
    ax3['wycinek'].set_title("Wycinek tęczy")
    plt.show()


