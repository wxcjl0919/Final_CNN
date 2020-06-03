import tensorflow as tf
import tensorflow.keras as ks
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials,space_eval
from sklearn.model_selection import StratifiedKFold
from utils import data_loader
from utils import mkdir
from utils import input_param
import numpy as np


def cnn_model_cross_valid(param):
    '''input : parameter of cnn model and the datasets'''

    kfold = StratifiedKFold(n_splits=2,shuffle=False)
    auc_score = []
    total_loss = []

    for train,test in kfold.split(datasets.X_train,datasets.Y_train):

        cnn_model = ks.Sequential()

        cnn_model.add(ks.layers.Conv1D(
            input_shape = (500,4),
            filters = param['conv1_filters'],
            kernel_size = param['conv1_kernel_size'],
            padding = 'same',
            activation = tf.nn.relu))

        cnn_model.add(ks.layers.MaxPool1D(
            pool_size=param['pool1_size'],
            strides= param['pool1_stride']))

        cnn_model.add(ks.layers.Conv1D(
            filters = param['conv2_filters'],
            kernel_size= param['conv2_kernel_size'],
            padding= 'same',
            activation= tf.nn.relu))

        cnn_model.add(ks.layers.MaxPool1D(
            pool_size=param['pool2_size'],
            strides= param['pool2_stride']))

        cnn_model.add(ks.layers.Flatten())

        cnn_model.add(ks.layers.Dense(
            units=param['dense1'],
            activation=tf.nn.relu))

        cnn_model.add(ks.layers.Dense(
            units= 1,
            activation=tf.nn.sigmoid,
            name = 'visualized_layer'))

        cnn_model.compile(
            optimizer=ks.optimizers.Adam(learning_rate= 0.0001,decay=1e-6),
            loss = ks.losses.BinaryCrossentropy(),
            metrics=[ks.metrics.AUC()])

        cnn_model.fit(
            x=datasets.X_train[train],
            y=datasets.Y_train[train],
            batch_size=param['batch_size'],
            epochs=param['num_epoch'],
            verbose=0)

        loss,auc = cnn_model.evaluate(datasets.X_train[test],datasets.Y_train[test],verbose = 0)
        total_loss.append(loss)
        auc_score.append(auc)
        cnn_model.summary()

    return total_loss,auc_score

def best_cnn_model(param):

    cnn_model = ks.Sequential()
    cnn_model.add(ks.layers.Conv1D(
        input_shape=(500, 4),
        filters=param['conv1_filters'],
        kernel_size=param['conv1_kernel_size'],
        padding='same',
        activation=tf.nn.relu))

    cnn_model.add(ks.layers.MaxPool1D(
        pool_size=param['pool1_size'],
        strides=param['pool1_stride']))

    cnn_model.add(ks.layers.Conv1D(
        filters=param['conv2_filters'],
        kernel_size=param['conv2_kernel_size'],
        padding='same',
        activation=tf.nn.relu))

    cnn_model.add(ks.layers.MaxPool1D(
        pool_size=param['pool2_size'],
        strides=param['pool2_stride']))

    cnn_model.add(ks.layers.Flatten())

    cnn_model.add(ks.layers.Dense(
        units=param['dense1'],
        activation=tf.nn.relu))
    cnn_model.add(ks.layers.Dense(
        units=1,
        activation=tf.nn.sigmoid,
        name='visualized_layer'))

    cnn_model.compile(
        optimizer=ks.optimizers.Adam(learning_rate=0.0001, decay=1e-6),
        loss= ks.losses.BinaryCrossentropy(),
        metrics=[ks.metrics.AUC()])

    history = cnn_model.fit(
        x=datasets.X_train,
        y=datasets.Y_train,
        batch_size=param['batch_size'],
        epochs=param['num_epoch'],
        validation_split=0.1,
        verbose=1)

    return cnn_model,history






def hyperopt(param):
    "get the best parameters by hyperopt"

    total_loss, _ = cnn_model_cross_valid(param)
    mean_loss = np.mean(total_loss)
    print(mean_loss)

    return {'loss':  mean_loss, 'status': STATUS_OK}


def best_param_hyperopt():
    '''run the hyperopy to get the best parameter'''
    '''output :  best parameter of the model'''

    space = {
        'conv1_filters': hp.choice('conv1_filters', [20,30,40]),
        'conv1_kernel_size': hp.choice('conv1_kernel_size', [5,10,15,20,25]),
        'pool1_size': hp.choice('pool1_size', [5,10,15,20]),
        'pool1_stride': hp.choice('pool1_stdride', [5,10,15,20]),
        'conv2_filters': hp.choice('conv2_filters', [50,60,70]),
        'conv2_kernel_size': hp.choice('conv2_kernel_size', [5,10,15,20,25,30]),
        'pool2_size': hp.choice('pool2_size', [3,5,7]),
        'pool2_stride': hp.choice('pool2_stdride', [1,5]),
        'dense1': hp.choice('dense1', [600,800,1000,1200,1400]),
        'batch_size': hp.choice('batch_size', [10, 15, 20,25,30]),
        'num_epoch': hp.choice('num_epoch', [30,40,50])}

    trails = Trials()
    best = fmin(hyperopt,space,algo=tpe.suggest,max_evals=3,trials= trails)

    best_param = space_eval(space,best)

    return best_param


def best_param_cnn(best_param,save_f,dataset_name):
    '''run the model in the best parameters and save the parameter and the performance'''
    '''output : the cnn model'''


    model,history = best_cnn_model(best_param)

    save_f.write('The model information of  ' + dataset_name + '\n')
    save_f.write('\n')

    save_f.write('#The parameter of Convolutional Neural Networks \n')
    for x in best_param.keys():
        save_f.write(x + ':    ' + str(best_param[x]) + '\n')

    save_f.write('\n')
    save_f.write('#The performance of model \n')
    for i in history.history.keys():
        save_f.write(i + ':    ' + str(history.history[i][-1]) + '\n')

    test_loss, test_auc = model.evaluate(datasets.X_test,datasets.Y_test)

    print(test_loss, test_auc)
    save_f.write('test loss:    ' + str(test_loss) + '\n')
    save_f.write('test auc:     ' + str(test_auc) + '\n')

    return model


def main():

    dataset_name = 'K562_USP_UUE'
    dataset_path = '/Users/xinzeng/Desktop/research/reproduce/'

    global datasets

    # input data from preproces
    datasets = data_loader(dataset_path + dataset_name+'.h5')

    # check whether the saved file exist!

    date = '5_4_2020'
    save_file_path = dataset_path + date
    mkdir(save_file_path)
    save_file_path = save_file_path + '/' + dataset_name
    mkdir(save_file_path)

    best_hyperparameters = input_param(dataset_path + 'K562_USP_UUE_best_model_info')

    # saved the hyperparameters of the model
    save_f = open(save_file_path + '/best_model_info', 'w')

    #run hyperopt to get the best hyperparameters

    print(best_hyperparameters)

    #run the best parameters on the model
    model = best_param_cnn(best_hyperparameters,save_f,dataset_name)

    model.save(save_file_path + '/best_model.h5')

    save_f.close()

if __name__ == '__main__':
    main()







