
import h5py
import numpy as np
import tensorflow as tf
import os

def convert_to_dna(one_hot_matrix):
    "convert one hot encoding to dna sequence "

    dna_sequence = ''

    for x in range(one_hot_matrix.shape[0]):

        if one_hot_matrix[x][0] == 1 :
            dna_sequence += 'A'
        if one_hot_matrix[x][1] == 1 :
            dna_sequence += 'C'
        if one_hot_matrix[x][2] == 1 :
            dna_sequence += 'G'
        if one_hot_matrix[x][3] == 1 :
            dna_sequence += 'T'

    return dna_sequence

def mkdir(path):

    folder = os.path.exists(path)

    if not folder :
        os.mkdir(path)
    else :
        print(path+ '  exist!')

class data_loader():

    def __init__(self,data_path):
        """import train,valid,test dataset"""

        dataset = h5py.File(data_path, 'r')

        self.X_train = np.array(dataset['X_train']).astype(np.float32)
        self.Y_train = np.array(dataset['Y_train']).astype(np.float32)
        self.X_test = np.array(dataset['X_test']).astype(np.float32)
        self.Y_test = np.array(dataset['Y_test']).astype(np.float32)

        self.X_train = self.X_train.transpose([0,2,1])
        self.X_test =  self.X_test.transpose([0,2,1])

        self.Y_train = self.Y_train.flatten()
        self.Y_test = self.Y_test.flatten()

        #to categorical
        #self.Y_train = tf.keras.utils.to_categorical(self.Y_train)
        #self.Y_test = tf.keras.utils.to_categorical(self.Y_test)

        #get the number of data
        self.num_train_data = self.X_train.shape[0]
        self.num_test_data = self.X_test.shape[0]


def input_param(file_apth):

    file_list = []
    file_dir = {}
    f = open(file_apth,'r')

    for x in f :
        file_list.append(x.strip())
    file_list = file_list[3:14]

    for x in file_list:
        x = x.split(':')

        file_dir[x[0]] = int(x[1].strip())

    return file_dir





