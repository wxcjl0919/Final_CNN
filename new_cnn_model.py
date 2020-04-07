import tensorflow as tf
import tensorflow.keras as ks
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials,space_eval
from sklearn.metrics import mean_absolute_error
from cnn.generate_data import data_loader
from cnn.generate_data import mkdir
import numpy as np

space = {
    'conv1_filters' : hp.choice('conv1_filters',[20,30,40]),
    'conv1_kernel_size': hp.choice('conv1_kernel_size',[5,10,15,20,25,30]),
    'pool1_size' : hp.choice('pool1_size',[5,10,15,20,25,30]),
    'pool1_stride': hp.choice('pool1_stdride',[1,10,20,30]),
    'conv2_filters': hp.choice('conv2_filters',[50,60,70]),
    'conv2_kernel_size': hp.choice('conv2_kernel_size',[5,10,15,20,25,30]),
    'pool2_size': hp.choice('pool2_size',[5,10,15]),
    'pool2_stride': hp.choice('pool2_stdride',[1,5,10]),
    'dense1' :hp.choice('dense1',[600,800,1000,1200,1400]),
    'batch_size' : hp.choice('batch_size',[10,15,20,25]),
    'num_epoch' : hp.choice('num_epoch',[30,40,50])}

def cnn_model(param,datasets):
    '''input : parameter of cnn model and the datasets'''

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
        units= 2,
        activation=tf.nn.softmax,
        name = 'visualized_layer'))

    cnn_model.compile(
        optimizer=ks.optimizers.Adam(learning_rate= 0.0001,decay=1e-6),
        loss = ks.losses.CategoricalCrossentropy(),
        metrics=[ks.metrics.AUC()])

    history = cnn_model.fit(

        x=datasets.X_train,
        y=datasets.Y_train,
        batch_size=param['batch_size'],
        epochs=param['num_epoch'],
        validation_data=(datasets.X_valid, datasets.Y_valid), verbose=0)



    return cnn_model, history


def run_hyperopt(param):
    "get the best parameters by hyperopt"

    model, _ = cnn_model(param,datasets)

    preds = model.predict(datasets.X_valid)

    test_loss, _ = model.evaluate(datasets.X_test, datasets.Y_test, verbose= 0 )
    acc = ks.losses.categorical_crossentropy(y_true=datasets.Y_valid,y_pred= preds)
    acc = np.mean(acc)

    acc = test_loss * acc

    return {'loss': acc, 'status': STATUS_OK}


def best_param():
    '''run the hyperopy to get the best parameter'''
    '''output :  best parameter of the model'''

    trails = Trials()
    best = fmin(run_hyperopt,space,algo=tpe.suggest,max_evals=100,trials= trails)

    best_param = space_eval(space,best)

    return best_param


def best_param_cnn(best_param,save_f):
    '''run the model in the best parameters and save the parameter and the performance'''
    '''output : the cnn model'''

    model,history = cnn_model(best_param,datasets)

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

def save_file(dataset_name,date):
    '''To make the file path and the dataset path '''

    file_path = '/Users/fais/Desktop/researches/results/'
    data_path = '/Users/fais/Desktop/researches/results/2_26_2020/dataset/'

    datasets = data_loader(data_path + dataset_name + '.h5')
    save_file_path = file_path + date

    mkdir(save_file_path)

    save_file_path = save_file_path +'/' + dataset_name

    mkdir(save_file_path)

    return datasets,save_file_path


# change the parameters here to get the result
dataset_name = 'K562_UUP_UUE'
date = '3_25_2020'
datasets,save_path = save_file(dataset_name,date)
save_f = open(save_path + '/best_model_info', 'w')

model = best_param_cnn(best_param(),save_f)
model.save(save_path+'/best_model.h5')
save_f.close()
'''
model = best_param_cnn(best_param(),save_f)
model.save(save_path+'/best_model.h5')
save_f.close()




model = best_param_cnn(best_parametes,save_f)
model.save(save_path+'best_model.h5')
save_f.close()
'''









