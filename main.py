import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image


def show8(img1, img2, img3, img4, img5, img6, img7, img8, title=''):
    plt.subplot(2, 4, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplot(2, 4, 2)
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.title('Frequency filter low ' + title)
    plt.subplot(2, 4, 3)
    plt.imshow(img4, cmap='gray')
    plt.axis('off')
    plt.title('Spectrum result low ' + title)
    plt.subplot(2, 4, 4)
    plt.imshow(img5, cmap='gray')
    plt.axis('off')
    plt.title('Image result low ' + title)
    plt.subplot(2, 4, 5)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.title('Image spectrum')
    plt.subplot(2, 4, 6)
    plt.imshow(img6, cmap='gray')
    plt.axis('off')
    plt.title('Frequency filter high ' + title)
    plt.subplot(2, 4, 7)
    plt.imshow(img7, cmap='gray')
    plt.axis('off')
    plt.title('Spectrum result high ' + title)
    plt.subplot(2, 4, 8)
    plt.imshow(img8, cmap='gray')
    plt.axis('off')
    plt.title('Image result high ' + title)
    plt.show()


def open_image(filename):
    try:
        image = cv2.imread(filename, 0)
        return image
    except FileNotFoundError:
        print("Файл не найден")
        return


def frequency(image, d_0, n):
    p, q = image.shape[:2]
    d = np.zeros((p, q))
    lowPerfect = np.zeros((p, q))
    lowButterworth = np.zeros((p, q))
    lowGaussian = np.zeros((p, q))
    highPerfect = np.zeros((p, q))
    highButterworth = np.zeros((p, q))
    highGaussian = np.zeros((p, q))
    for u in range(q):
        for v in range(p):
            d[u][v] = ((u - p / 2) ** 2 + (v - q / 2) ** 2) ** 0.5
            lowPerfect[u][v] = (d[u][v] <= d_0)
            lowButterworth[u][v] = 1 / ((1 + d[u][v] / d_0) ** (2 * n))
            lowGaussian[u][v] = math.exp(-(d[u][v] * d[u][v]) / (2 * d_0 * d_0))

            highPerfect[u][v] = (d[u][v] <= d_0)
            highPerfect[u][v] = (d[u][v] > d_0)
            if d[u][v] != 0:
                highButterworth[u][v] = 1 / ((1 + d_0 / d[u][v]) ** (2 * n))
            else:
                highButterworth[u][v] = 0
            highGaussian[u][v] = 1 - lowGaussian[u][v]
    return lowPerfect, lowButterworth, lowGaussian, highPerfect, highButterworth, highGaussian


def perfect_filter(image, low_perfect, high_perfect):
    pow = 1
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    spectrum_log = np.log(np.abs(spectrum))

    fft_low = low_perfect * spectrum
    spectrum_low = (low_perfect * spectrum_log) ** pow
    result_low = np.fft.ifft2(np.fft.ifftshift(fft_low))
    result_low = np.abs(result_low)

    fft_high = high_perfect * spectrum
    spectrum_high = (high_perfect * spectrum_log) ** pow
    result_high = np.fft.ifft2(np.fft.ifftshift(fft_high))
    result_high = np.abs(result_high)

    show8(image, spectrum_log, low_perfect, spectrum_low, result_low,
          high_perfect, spectrum_high, result_high, 'Perfect')


def butterworth(image, low_butterworth, high_butterworth):
    pow = 0.4
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    spectrum_log = np.log(np.abs(spectrum))

    fft_low = low_butterworth * spectrum
    spectrum_low = (low_butterworth * spectrum_log) ** pow
    result_low = np.fft.ifft2(np.fft.ifftshift(fft_low))
    result_low = np.abs(result_low)

    fft_high = high_butterworth * spectrum
    spectrum_high = (high_butterworth * spectrum_log) ** pow
    result_high = np.fft.ifft2(np.fft.ifftshift(fft_high))
    result_high = np.abs(result_high)

    show8(image, spectrum_log, low_butterworth, spectrum_low, result_low,
          high_butterworth, spectrum_high, result_high, 'Butterworth')


def gaussian(image, low_gaussian, high_gaussian):
    pow = 0.5
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    spectrum_log = np.log(np.abs(spectrum))

    fft_low = low_gaussian * spectrum
    spectrum_low = (low_gaussian * spectrum_log) ** pow
    result_low = np.fft.ifft2(np.fft.ifftshift(fft_low))
    result_low = np.abs(result_low)

    fft_high = high_gaussian * spectrum
    spectrum_high = (high_gaussian * spectrum_log) ** pow
    result_high = np.fft.ifft2(np.fft.ifftshift(fft_high))
    result_high = np.abs(result_high)

    show8(image, spectrum_log, low_gaussian, spectrum_low, result_low,
          high_gaussian, spectrum_high, result_high, 'Gaussian')


if __name__ == '__main__':
    filename = "aaa.png"
    d_0 = 30
    n = 2
    image_original = open_image(filename)
    lowPerfect, lowButterworth, lowGaussian, highPerfect, highButterworth, highGaussian = \
        frequency(image_original, d_0, n)
    perfect_filter(image_original, lowPerfect, highPerfect)
    butterworth(image_original, lowButterworth, highButterworth)
    gaussian(image_original, lowGaussian, highGaussian)
