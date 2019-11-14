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
zeros = [
    0.9*np.exp(np.complex(0, np.pi/2)),
    0.9*np.exp(np.complex(0, -np.pi/2)),
    0.95*np.exp(np.complex(0, np.pi/8)),
    0.95*np.exp(np.complex(0, -np.pi/8))
]

poles = [
    0,
    -0.99,
    -0.99,
    0.8
]

angles = np.linspace(0,2*np.pi,1000)
plt.figure()
for zero in zeros:
    plt.scatter(np.real(zero), np.imag(zero), marker="o", c='b')
for pole in poles:
    plt.scatter(np.real(pole), np.imag(pole), marker="x", c='r')
plt.plot(np.cos(angles),np.sin(angles))
plt.show()


# Designing reverse filter for aberrations
########################################################################################################################
b = [1, -1.75537, 1.7125, -1.42185, 0.73105]
a = [1, 1.18, -0.6039, -0.78408, 0]



plt.figure()
x,y = signal.freqz(b,a)
plt.plot(x, 20*np.log10(abs(y)))
plt.show()


aberration = np.load('pictures/goldhill_aberrations.npy')

plt.figure()
output=signal.lfilter(a, b, aberration)
plt.imshow(output,cmap='gray')
plt.show()

########################################################################################################################