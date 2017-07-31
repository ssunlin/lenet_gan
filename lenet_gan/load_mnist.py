import os
import gzip
import numpy
import math
import scipy.misc

# mnist_path="E:\dataset\mnist"
train_images="train-images-idx3-ubyte.gz"
train_labels="train-labels-idx1-ubyte.gz"
test_images="t10k-images-idx3-ubyte.gz"
test_labels="t10k-labels-idx1-ubyte.gz"

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename, mnist_path):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
      f: A file object that can be passed into a gzip reader.
    Returns:
      data: A 4D uint8 numpy array [index, y, x, depth].
    Raises:
      ValueError: If the bytestream does not start with 2051.
    """
    if filename=="train_images":
        filename=os.path.join(mnist_path,train_images)
    elif filename=='test_images':
        filename=os.path.join(mnist_path,test_images)
    print("Extracting",filename)
    with gzip.GzipFile(filename=filename) as bytestream:
        magic=_read32(bytestream)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(filename, mnist_path):
    """Extract the labels into a 1D uint8 numpy array [index].
    Args:
      f: A file object that can be passed into a gzip reader.
      one_hot: Does one hot encoding for the result.
      num_classes: Number of classes for the one hot encoding.
    Returns:
      labels: a 1D uint8 numpy array.
    Raises:
      ValueError: If the bystream doesn't start with 2049.
    """
    if filename=='train_labels':
        filename=os.path.join(mnist_path,train_labels)
    elif filename=='test_labels':
        filename=os.path.join(mnist_path,test_labels)
    print("Extracting",filename)
    with open(filename,'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                                 (magic, f.name))
            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            labels = numpy.frombuffer(buf, dtype=numpy.uint8)
            return labels

def save_sample(sample,name):
    sp = sample.shape
    sample=numpy.reshape(sample,(sp[0],sp[1],sp[2]))
    h=int(math.sqrt(sp[0]))
    w=int(math.sqrt(sp[0]))
    image=numpy.zeros((h*sp[1],w*sp[2]))
    for i in range(h):
        for j in range(w):
            h1=i * sp[1]
            h2=(i+1)*sp[1]
            w1=j*sp[2]
            w2=(j+1)*sp[2]
            ind=i*h+j
            image[h1:h2,w1:w2]=sample[ind]
    scipy.misc.imsave(name,image)





