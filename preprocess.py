import numpy as np
from kmer import kmer_featurization
import h5py


def parse_sequences(seq_path):
    """Parse fasta file for sequences

    input fasta format file, output list of sequence"""
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
    """convert DNA/RNA sequences to a one-hot representation

    input a DNA sequence output a one-hot of sequence"""
    one_hot_seq = []
    for seq in sequences :
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4,seq_length))

    # The fisrt column is A, second is C, thrid is G, fourth is U or T
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


def k_mer_encoding(dna_sequences,k):

    obj = kmer_featurization(k)
    k_mer_features = obj.obtain_kmer_feature_for_a_list_of_sequences(dna_sequences)

    return k_mer_features


def split_dataset(one_hot,labels,test_frac=0.1):
    """split dataset into training, and test set

    input a dataset, output train,test set"""
    def split_index(num_data,test_frac):
        #generate the index by random

        train_frac = 1 - test_frac
        cum_index = np.array(np.cumsum([0,train_frac,test_frac])*num_data).astype(int)
        shuffle = np.random.permutation(num_data)
        train_index = shuffle[cum_index[0]:cum_index[1]]
        test_index =shuffle[cum_index[1]:cum_index[2]]

        return train_index,test_index

    num_data = len(one_hot)
    train_index,test_index = split_index(num_data,test_frac)

    #split dataset
    train = (one_hot[train_index],labels[train_index,:])
    test = (one_hot[test_index],labels[test_index,:])
    indices = [train_index,test_index]

    return train,test,indices



def main():

    data_file_path = '/Users/xinzeng/Desktop/research/result/5_27_2020/'
    pos = 'K562_US.fa'
    neg = 'K562_UU.fa'
    pos_sequence = parse_sequences(data_file_path + pos)
    neg_sequence = parse_sequences(data_file_path + neg)


    pos_one_hot = convert_one_hot(pos_sequence)
    neg_one_hot = convert_one_hot(neg_sequence)

    one_hot = np.vstack([pos_one_hot,neg_one_hot])
    labels = np.vstack([np.ones((len(pos_one_hot), 1)), np.zeros((len(neg_one_hot), 1))])

    train, test, indices = split_dataset(one_hot, labels, test_frac=0.1)


    # save dataset as hdf5 file
    with h5py.File(data_file_path + 'K562_US_UU_onehot.h5','w') as f :
        dset = f.create_dataset("X_train", data=train[0].astype(np.float32), compression="gzip")
        dset = f.create_dataset("Y_train", data=train[1].astype(np.float32), compression="gzip")
        dset = f.create_dataset("X_test", data=test[0].astype(np.float32), compression="gzip")
        dset = f.create_dataset("Y_test", data=test[1].astype(np.float32), compression="gzip")

    print('successfully save! ')



if __name__ == '__main__':
    main()


"""
    pos_k_mer = k_mer_encoding(pos_sequence,6)
    neg_k_mer = k_mer_encoding(neg_sequence,6)

    k_mer = np.vstack([pos_k_mer,neg_k_mer])
    labels = np.vstack([np.ones((len(pos_k_mer), 1)), np.zeros((len(neg_k_mer), 1))])
"""
