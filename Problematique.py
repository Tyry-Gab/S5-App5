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
    angles = np.linspace(0,2*np.pi,1000)
    plt.figure()
    for zero in zeros:
        plt.scatter(np.real(zero), np.imag(zero), marker="o", c='b')
    for pole in poles:
        plt.scatter(np.real(pole), np.imag(pole), marker="x", c='r')
    plt.plot(np.cos(angles),np.sin(angles))
    plt.show()


def transfer_from_poles_zeros(poles,zeros,show=False):
    b = np.poly(zeros)
    a = np.poly(poles)

    if show:
        plt.figure()
        x,y = signal.freqz(b,a)
        plt.plot(x, 20*np.log10(abs(y)))
        plt.show()
    return b,a


def remove_aberrations(poles,zeros, source_image):
    aberration = np.load(source_image)
    b,a = transfer_from_poles_zeros(poles,zeros)
    plt.figure()
    output=signal.lfilter(a, b, aberration)
    plt.imshow(output,cmap='gray')
    plt.show()
    return output


########################################################################################################################
if __name__ == "__main__":
    cleaned = remove_aberrations(Poles,Zeros,'pictures/goldhill_aberrations.npy')
    scipy.misc.imsave('pictures/main/aberrations.jpg', cleaned)

