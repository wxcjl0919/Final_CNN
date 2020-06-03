import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from hyperopt import tpe,STATUS_OK,Trials,space_eval

from hyperas import optim
from hyperas.distributions import choice
import h5py




def load_data():

    dataset_name = 'K562_US_UU_onehot'
    dataset_path = '/Users/xinzeng/Desktop/research/result/5_27_2020/'
    data_path = dataset_path + dataset_name + '.h5'
    dataset = h5py.File(data_path, 'r')

    X_train = np.array(dataset['X_train']).astype(np.float32)
    Y_train = np.array(dataset['Y_train']).astype(np.float32)
    X_test = np.array(dataset['X_test']).astype(np.float32)
    Y_test = np.array(dataset['Y_test']).astype(np.float32)

    X_train = X_train.transpose([0, 2, 1])
    X_test = X_test.transpose([0, 2, 1])

    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    print('successfully load the data')

    return X_train,Y_train,X_test,Y_test


def cnn_model(X_train,Y_train,X_test,Y_test):

    model = ks.Sequential()

    model.add(ks.layers.Conv1D(
        input_shape = (500,4),
        filters = {{choice([20, 30, 40])}},
        kernel_size = {{choice([5, 10, 15, 20, 25])}},
        padding = 'same',
        activation = tf.nn.relu))

    model.add(ks.layers.MaxPool1D(
        pool_size = {{choice([5, 10, 15, 20])}},
        strides = {{choice([5, 10, 15, 20])}}))

    model.add(ks.layers.Dropout(0.5))

    model.add(ks.layers.Conv1D(
        filters={{choice([50, 60, 70])}},
        kernel_size={{choice([5, 10, 15, 20, 25, 30])}},
        padding='same',
        activation=tf.nn.relu))

    model.add(ks.layers.MaxPool1D(
        pool_size={{choice([3, 5, 7])}},
        strides={{choice([1, 5])}}))

    model.add(ks.layers.Dropout(0.5))

    model.add(ks.layers.Flatten())

    model.add(ks.layers.Dense(
        units={{choice([600, 800, 1000, 1200, 1400])}},
        activation=tf.nn.relu))

    model.add(ks.layers.Dense(
        units=1,
        activation=tf.nn.sigmoid,
        name='visualized_layer'))

    model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=0.0001, decay=1e-6),
        loss= ks.losses.BinaryCrossentropy(),
        metrics=[ks.metrics.AUC()])

    result = model.fit(X_train, Y_train,
              batch_size={{choice([10,15,20,25,30])}},
              epochs={{choice([30,40,50])}},
              verbose=0,
              validation_split=0.1)

    validation_loss = np.amin(result.history['loss'])
    print('Best validation loss of epoch:', validation_loss)

    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}

def write_model_information(best_run,save_file):

    save_file.write('The model information of  ' + dataset_name + '\n')
    save_file.write('\n')

    for x in best_run.keys():
        save_file.write(x + ':    ' + str(best_run[x]) + '\n')







def main():

    global dataset_name
    global dataset_path

    dataset_name = 'K562_US_UU_onehot'
    dataset_path = '/Users/xinzeng/Desktop/research/result/5_27_2020/'

    X_train, Y_train, X_test, Y_test = load_data()


    best_run, best_model = optim.minimize(
        model = cnn_model,
        data = load_data,
        algo = tpe.suggest,
        max_evals = 100,
        eval_space=True,
        trials= Trials())

    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    save_file_path = '/Users/xinzeng/Desktop/research/result/5_29_2020/K562_US_UU/'
    save_info = open(save_file_path + 'best_model_info', 'w')

    write_model_information(best_run,save_info)
    best_model.save(save_file_path + 'best_model.h5')

    save_info.close()

if __name__ == '__main__':
    main()
