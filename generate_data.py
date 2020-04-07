import numpy as np
import h5py
import tensorflow as tf
import os
def parse_sequences(seq_path):
    """Parse fasta file for sequences"""

    f = open(seq_path,'r')
    seq = {}
    for line in f :
        if line.startswith('>') :
            name = line.replace('>','').split()[0]
            seq[name] = ''
        else :
            seq[name] += line.replace('\n','').strip()

    sequences = list(seq.values())
    sequences = np.array(sequences)
    f.close()

    return sequences


def convert_one_hot(sequences):
    """convert DNA/RNA sequences to a one-hot representation"""

    one_hot_seq = []
    for seq in sequences :
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4,seq_length))

        index = [j for j in range(seq_length) if seq[j] == 'A']
        one_hot[0,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'C']
        one_hot[1, index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'G']
        one_hot[2, index] = 1
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        one_hot[3, index] = 1

        one_hot_seq.append(one_hot)

    #convert to numpy array
    one_hot_seq = np.array(one_hot_seq)

    return one_hot_seq


def split_dataset(one_hot,labels,valid_frac=0.1,test_frac=0.1):
    """split dataset into training, cross-validation, and test set"""

    def split_index(num_data,valid_frac,test_frac):
        #generate the index by random

        train_frac = 1 - valid_frac - test_frac
        cum_index = np.array(np.cumsum([0,train_frac,valid_frac,test_frac])*num_data).astype(int)
        shuffle = np.random.permutation(num_data)
        train_index = shuffle[cum_index[0]:cum_index[1]]
        valid_index = shuffle[cum_index[1]:cum_index[2]]
        test_index =shuffle[cum_index[2]:cum_index[3]]

        return train_index,valid_index,test_index

    num_data = len(one_hot)
    train_index,valid_index,test_index = split_index(num_data,valid_frac,test_frac)

    #split dataset
    train = (one_hot[train_index],labels[train_index,:])
    valid = (one_hot[valid_index],labels[valid_index,:])
    test = (one_hot[test_index],labels[test_index,:])
    indices = [train_index,valid_index,test_index]

    return train,valid,test,indices


def generate_dataset(pos_filename,neg_filename):
    '''generate training set, validation set and independent set'''

    pos_sequence = parse_sequences(file_path + pos_filename)
    neg_sequence = parse_sequences(file_path + neg_filename)
    pos_one_hot = convert_one_hot(pos_sequence)
    neg_one_hot = convert_one_hot(neg_sequence)

    one_hot = np.vstack([pos_one_hot,neg_one_hot])
    labels = np.vstack([np.ones((len(pos_one_hot),1)),np.zeros((len(neg_one_hot),1))])
    train,valid,test,indices = split_dataset(one_hot,labels,valid_frac=0.1,test_frac=0.1)

    return train,valid,test,indices


def sava_dataset(pos_filename,neg_filename,save_path):
    """save datasets as hdf5 file"""
    print('saving dataset to ' + save_path)

    train,valid,test,_ = generate_dataset(pos_filename,neg_filename)

    with h5py.File(save_path,'w') as f :
        dset = f.create_dataset("X_train", data=train[0].astype(np.float32), compression="gzip")
        dset = f.create_dataset("Y_train", data=train[1].astype(np.float32), compression="gzip")
        dset = f.create_dataset("X_valid", data=valid[0].astype(np.float32), compression="gzip")
        dset = f.create_dataset("Y_valid", data=valid[1].astype(np.float32), compression="gzip")
        dset = f.create_dataset("X_test", data=test[0].astype(np.float32), compression="gzip")
        dset = f.create_dataset("Y_test", data=test[1].astype(np.float32), compression="gzip")


    print('successfully save! ')

class data_loader():

    def __init__(self,data_path):
        """import train,valid,test dataset"""

        dataset = h5py.File(data_path, 'r')

        self.X_train = np.array(dataset['X_train']).astype(np.float32)
        self.Y_train = np.array(dataset['Y_train']).astype(np.float32)
        self.X_valid = np.array(dataset['X_valid']).astype(np.float32)
        self.Y_valid = np.array(dataset['Y_valid']).astype(np.float32)
        self.X_test = np.array(dataset['X_test']).astype(np.float32)
        self.Y_test = np.array(dataset['Y_test']).astype(np.float32)

        #add another dimenson to make it a 4D tensor
        self.X_train = self.X_train.transpose([0,2,1])
        self.X_valid = self.X_valid.transpose([0,2,1])
        self.X_test =  self.X_test.transpose([0,2,1])

        #to categorical
        self.Y_train = tf.keras.utils.to_categorical(self.Y_train)
        self.Y_valid = tf.keras.utils.to_categorical(self.Y_valid)
        self.Y_test = tf.keras.utils.to_categorical(self.Y_test)

        #get the number of data
        self.num_train_data = self.X_train.shape[0]
        self.num_valid_data = self.X_valid.shape[0]
        self.num_test_data = self.X_test.shape[0]

    def get_batch(self,batch_size):

        index = np.random.randint(0,np.shape(self.X_train)[0],batch_size)

        return self.X_train[index,:],self.Y_train[index]



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








'''
file_path = '/Users/fais/Desktop/researches/results/1_11_2020/'
pos= 'gm12878US_P.fa'
neg = 'gm12878UU_P.fa'
save_path = '/Users/fais/Desktop/researches/results/2_26_2020/'
save_path = save_path + 'GM12878USP_UUP.h5'

sava_dataset(pos,neg,save_path)

'''



