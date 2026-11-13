import numpy as np # linear algebra
import struct
import torch
from array import array
from os.path  import join

class MnistDataloader(object):
    def __init__(self, path):
        self.training_images_filepath = path + "train-images.idx3-ubyte" #training_images_filepath
        self.training_labels_filepath = path + "train-labels.idx1-ubyte" #training_labels_filepath
        self.test_images_filepath = path + "t10k-images.idx3-ubyte" #test_images_filepath
        self.test_labels_filepath = path + "t10k-labels.idx1-ubyte" #test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath, reshape = False):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049: raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051: raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            if reshape: img = img.reshape(rows, cols)
            images.append(img)
        
        return np.array(images), torch.tensor(labels, dtype=torch.long)
            
    def load_data(self, reshape = False):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath, reshape)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath, reshape)
        return (x_train, y_train), (x_test, y_test)        