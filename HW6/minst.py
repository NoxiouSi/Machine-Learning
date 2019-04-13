import os, struct
from array import array as pyarray
import numpy as np

# data from  http://yann.lecun.com/exdb/mnist/
# sliced from here http://g.sweyla.com/blog/2012/mnist-numpy/

def load_mnist(dataset="training", digits=np.arange(10), path='.'):
    fname_img = os.path.join(path, 'train-images.idx3-ubyte')
    fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    # ind = [ k for k in range(size) ]
    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = np.zeros((N, rows, cols))
    labels = np.zeros(N)
    for i in range(len(ind)):
        images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    return images, labels
