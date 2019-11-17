import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import signal
import zplane
from PIL import Image
import scipy.misc

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


def remove_aberrations(poles,zeros, source_image, show=False):
    aberration = np.load(source_image)
    b,a = transfer_from_poles_zeros(poles,zeros)
    output = signal.lfilter(a, b, aberration)
    if show:
        plt.figure()
        plt.imshow(output,cmap='gray')
        plt.show()
    return output


def rotate_image(source_image, npy=False, show=False):
    img = np.empty(shape=[0,0])
    if npy:
        img = np.load(source_image)
    else:
        img = np.array(Image.open(source_image))

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
            # For now, only gray picture will be used (we assume input and output are gray)
            if type(elem) is np.ndarray:
                elem = elem[0]
            new_img[int(round(new_index[0][0]))][int(round(new_index[1][0]))] = elem
            y += 1
        x += 1

    if show:
        plt.figure()
        plt.gray()
        plt.imshow(new_img)
        plt.show()

    return new_img


def noise_removal_bilinear(source_image, npy=True, show=False):
    if npy:
        source_image = np.load(source_image)
    butter_order = 2
    Fp = 650
    Fc = 750
    Fe = 1600
    T = 1/Fe

    wc = Fe*np.tan(np.pi*Fp/Fe)

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

    order, Wn = signal.buttord(Fp/Fe, Fc/Fe, gpass=0.5, gstop=40)
    print(order)
    b, a, *_ = signal.butter(order,Wn)

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



########################################################################################################################
if __name__ == "__main__":

    cleaned = remove_aberrations(Poles,Zeros,'pictures/goldhill_aberrations.npy', show=True)
    scipy.misc.imsave('pictures/main/aberrations.jpg', cleaned)

    noise_bilinear = noise_removal_bilinear('pictures/goldhill_bruit.npy', show=True)
    noise_cheat = noise_removal_cheat(cleaned, npy=False, show=True)
    #Returns the np.array of the rotated picture
    rotate_image('pictures/goldhill_rotate_source.png', show=True)


    scipy.misc.imsave('pictures/main/noise.jpg', noise)

    rotate_image(None)





