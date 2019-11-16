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


def rotate_image(source_image):
    pass

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
    cleaned = remove_aberrations(Poles,Zeros,'pictures/image_complete.npy', show=True)
    scipy.misc.imsave('pictures/main/aberrations.jpg', cleaned)

    rotate_image()




