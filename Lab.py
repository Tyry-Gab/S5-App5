
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import signal
import zplane
from PIL import Image
import scipy.misc


# Num 1:
K = 1
z1=np.complex(0,0.8)
z2=np.complex(0,-0.8)
p1=0.95*np.exp(np.complex(0,np.pi/8))
p2=0.95*np.exp(np.complex(0,-np.pi/8))

angles = np.linspace(0,2*np.pi,1000)

fig = plt.figure()
plt.scatter(z1.real, z1.imag)
plt.scatter(z2.real,z2.imag)
plt.scatter(p1.real,p1.imag)
plt.scatter(p2.real,p2.imag)
plt.plot(np.cos(angles),np.sin(angles))
plt.show()



b = [1,(-z1-z2),z1*z2]
a = [1,(-p1-p2),p1*p2]
H_z = K*np.poly(b)/np.poly(a)


plt.figure()
x,y = signal.freqz(b,a,whole=True)
plt.plot(x, 20*np.log10(abs(y)))
plt.show()

size = 101
pulse = []
for i in range(size):
    pulse.append(0)
pulse[int((size-1)/2)]=1

plt.figure()
output = signal.lfilter(b,a,pulse)
plt.plot(output)
plt.show()

plt.figure()
plt.plot(pulse)
output = signal.lfilter(a,b,output)
plt.plot(output)
plt.show()


# Num 2

b = (1, -2*np.cos(np.pi/16), 1)
a = (1, -2*0.90*np.cos(np.pi/16), 0.90**2)

plt.figure()
x,y = signal.freqz(b,a,whole=True)
plt.plot(x,y)
plt.show()

sig=[]
size=500
for i in range(size):
    sig.append(np.sin(i*np.pi/16) + np.sin(i*np.pi/32))

plt.figure()
output=signal.lfilter(b,a,sig)
plt.plot(sig)
plt.plot(output)
plt.show()


# Num 3

order, Wn = signal.buttord(2.5/24, 3.5/24, gpass=0.2, gstop=40)
print(order)
b,a,*_ = signal.butter(order, Wn)

zplane.zplane(b,a)

plt.figure()
x,y = signal.freqz(b,a)
plt.plot(x, 20*np.log10(abs(y)))
plt.show()

# Num 4


T = np.matrix(
    [[2,0],
    [0,0.5]]
)

a = np.matrix(
    [[3, 1],
     [4, 7]]
)

res = a*T
print(res)

img = Image.open('pictures/goldhill.png')
array = np.array(img)


elongated_array = np.array([[0 for x in range(1024)] for y in range(256)])

print(elongated_array)
y=0
x=0
for row in array:
    x=0
    for value in row:
        new_val = T*np.array([
            [x],
            [y]
        ])
        if new_val[0] % 2 ==0:
            elongated_array[int(new_val[1])][int(new_val[0])] = value
        x+=1
    y+=1

np.set_printoptions(threshold=np.inf)
f1 = open('pictures/Lab1-n4.txt', 'w+')
f1.write(str(elongated_array))
f1.close()
f1 = open('pictures/goldhill.txt', 'w+')
f1.write(str(array))
f1.close()

plt.figure()
plt.gray()
plt.imshow(elongated_array)
plt.show()

img=Image.fromarray(elongated_array,mode='I')
img.save('pictures/Lab1-n4.png')
scipy.misc.imsave('outfile.jpg', elongated_array)





