import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import signal
import zplane
from PIL import Image
import scipy.misc
import imageio


# Getting poles and zeros from aberrations transfer function
########################################################################################################################

Zeros = [
    0.9*np.exp(np.complex(0, np.pi/2)),
    0.9*np.exp(np.complex(0, -np.pi/2)),
    0.95*np.exp(np.complex(0, np.pi/8)),
    0.95*np.exp(np.complex(0, -np.pi/8))
]

Poles = [
    0,
    -0.99,
    -0.99,
    0.8
]

# Functions for operations
########################################################################################################################

def scatter_poles_zeros(poles, zeros):
    b = np.poly(zeros)
    a = np.poly(poles)
    zplane.zplane(b,a)


def transfer_from_poles_zeros(poles,zeros,show=False):
    b = np.poly(zeros)
    a = np.poly(poles)

    if show:
        plt.figure()
        x,y = signal.freqz(b,a)
        plt.plot(x, 20*np.log10(abs(y)))
        plt.show()
    return b,a


def remove_aberrations(poles,zeros, source_image, npy=False, show=False):
    if npy:
        source_image = np.load(source_image)
    b,a = transfer_from_poles_zeros(poles,zeros)
    output = signal.lfilter(a, b, source_image)
    if show:
        plt.figure()
        plt.imshow(output,cmap='gray')
        plt.show()
    return output


def rotate_image(source_image, npy=False, png=False, show=False):
    img = np.empty(shape=[0,0])
    if npy:
        img = np.load(source_image)
    elif png:
        img = np.array(Image.open(source_image).convert('L'))
    else:
        img = source_image

    new_img = np.empty(shape=(img.shape[1], img.shape[0]))

    rot_angle = -np.pi/2
    rot_mat = np.array(
        [[np.cos(rot_angle), -np.sin(rot_angle)],
        [np.sin(rot_angle), np.cos(rot_angle)]])
    x = 1-img.shape[0]
    for row in img:
        y = 0
        for elem in row:
            new_index = rot_mat.dot(np.array([[x],[y]]))
            new_img[int(round(new_index[0][0]))][int(round(new_index[1][0]))] = elem
            y += 1
        x += 1

    if show:
        plt.figure()
        plt.gray()
        plt.title('Image avant rotation')
        plt.imshow(source_image)
        plt.show()

        plt.figure()
        plt.gray()
        plt.title('Image après rotation (90 degrés vers la droite)')
        plt.imshow(new_img)
        plt.show()

    return new_img


def noise_removal_bilinear(source_image, npy=True, show=False):
    if npy:
        source_image = np.load(source_image)
    Fc = 650
    Fe = 1600
    T = 1/Fe

    wc = Fe*np.tan(np.pi*Fc/Fe)

    b = [
        T**2*wc**2,
        2*T**2*wc**2,
        T**2*wc**2
    ]
    a = [
        (4 + 2*np.sqrt(2)*T*wc + T**2*wc**2),
        (-8 + T**2*wc**2),
        (4 -np.sqrt(2)*2*T*wc + T**2*wc**2)
    ]
    a = np.asarray(a)
    b = np.asarray(b)
    print(a)
    print(b)
    if show:
        plt.figure()
        x,y = signal.freqz(b,a)
        plt.plot(x, 20*np.log10(abs(y)))
        plt.show()
        print(a)
        print(b)
        zplane.zplane(b,a)

    output = signal.lfilter(b, a, source_image)

    if show:
        plt.figure()
        plt.figure()
        plt.imshow(output, cmap='gray')
        plt.show()
    return output


def noise_removal_cheat(source_image, npy=True, show=False):
    if npy:
        source_image = np.load(source_image)
    Fp = 650
    Fc = 750
    Fe = 1600

    order, Wn = signal.buttord(2*Fp/Fe, 2*Fc/Fe, gpass=0.5, gstop=40)
    print(order)
    #order = 2
    b, a, *_ = signal.butter(order, Wn)

    if show:
        plt.figure()
        x, y = signal.freqz(b, a)
        plt.plot(x, 20 * np.log10(abs(y)))
        plt.show()

    output = signal.lfilter(b, a, source_image)

    if show:
        plt.figure()
        plt.figure()
        plt.imshow(output, cmap='gray')
        plt.show()
    return output


def compress(image_array, show=False, pourcent=0.5):
    covariance_matrix = np.cov(image_array)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    compressed_img = eigenvectors.dot(image_array)

    if show:
        plt.figure()
        plt.imshow(compressed_img, cmap='gray')
        plt.show()

    k = int(round(pourcent * compressed_img.shape[0]))
    abs100_compressed_image = 100*np.abs(compressed_img)
    ind = np.argpartition(abs100_compressed_image.T[0], k)[:k]
    i = 0
    for index in ind:
        compressed_img[index].fill(0)
        i += 1
    print("There were {0} lines removed".format(i))

    return compressed_img, eigenvectors


def decompress(cm_img, eig_vectors, pourcent=0.0, show=False):
    decompressed = np.linalg.inv(eig_vectors).dot(cm_img)
    if show:
        plt.figure()
        plt.title('Image compressée à {0}%'.format(100*pourcent))
        plt.imshow(decompressed, cmap='gray')
        plt.show()
    return decompressed


########################################################################################################################
if __name__ == "__main__":

    # Cleaning up image
    cleaned = remove_aberrations(Poles,Zeros,'pictures/image_complete.npy', npy=True)
    turned = rotate_image(cleaned, npy=False, show=True)
    final = noise_removal_bilinear(turned,npy=False, show=True)
    imageio.imwrite('pictures/main/cleaned_up.jpg', final)

    compressed50, vectors50 = compress(final, pourcent=0.5, show=False)
    compressed75, vectors75 = compress(final, pourcent=0.75, show=False)
    guess_whos_back50 = decompress(compressed50, vectors50, pourcent=0.5, show=True)
    guess_whos_back75 = decompress(compressed75, vectors75, pourcent=0.75, show=True)

    turned_goldhill = rotate_image('pictures/goldhill_rotate_source.png', png=True)
    imageio.imwrite('pictures/main/goldhill_rotate_right.jpg', turned_goldhill)
    imageio.imwrite('pictures/main/miaow_compressed50.jpg', guess_whos_back50)







