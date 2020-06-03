
from tensorflow.keras.models import load_model
from vis.visualization import visualize_saliency
from vis.utils import utils
import numpy as np
import keras
import seaborn as sns
import matplotlib.pyplot as plt

from utils import convert_to_dna
from utils import data_loader


def best_predict_sequence(predict_result,num_best_sequence,pred_class):

    best_result = np.round(predict_result, decimals=3)[:,pred_class]

    best_index = np.argsort(-best_result)[0:num_best_sequence]

    return best_index

def total_saliency_map(model,datasets,lay_index,index,filter_indices):

    total_grad = np.zeros(datasets.X_test.shape[1])

    for x in range(len(index)):

        _,grads = visualize_saliency(model= model,
                                     layer_idx= lay_index,
                                     filter_indices = filter_indices,
                                     seed_input= datasets.X_test[index[x]])

        grads = grads/np.sum(grads)
        total_grad = np.vstack((total_grad,grads))

    total_grad = np.delete(total_grad,0,axis=0)
    total_grad = np.round(total_grad, decimals=3)

    return total_grad

def grad_heatmap(grads):

     f, ax = plt.subplots(figsize=(20, 3))
     sns.heatmap(grads, cmap='seismic', xticklabels=50,yticklabels=5,center=0.002)
     ax.set_title('Saliency map of top 50 predicted promoters in Promoter-Enhancer CNN model')
     plt.show()


def lineplot(promoters_enhancer,promoter_random):

    random = np.ones(500) / 500
    x = np.arange(0, 500)

    plt.plot(x,promoters_enhancer,color = 'red',label = 'promoters-enhancer')
    plt.plot(x, promoter_random, color='blue', label='promoter-random')

    plt.plot(x,random,color = 'grey',label = 'avergae importance')
    plt.yticks(np.arange(0,0.008,0.002))
    plt.xticks(np.arange(0,500,50))
    plt.xlabel('positions')
    plt.ylabel('importance')
    plt.title('The mean importance of each position in top 50 predicted promoters')

    plt.legend()
    plt.show()

def main():

    # the file path of data and model
    dataset_dir = '/Users/xinzeng/Desktop/research/reproduce/'
    dataset_name1 = 'K562_USP_UUE'
    dataset_name2 = 'K562_US_Promoters_Cnotrol'
    date ='5_4_2020'

    dataset1 = data_loader( dataset_dir + dataset_name1 +'.h5')
    dataset2 =  data_loader( dataset_dir + dataset_name2 +'.h5')

    cnn_model1 = load_model(dataset_dir + date +'/' + dataset_name1 + '/best_model.h5')
    cnn_model2 = load_model(dataset_dir + date + '/' + dataset_name2 + '/best_model.h5')

    test_result1 = cnn_model1(dataset1.X_test)
    test_result2 = cnn_model2(dataset2.X_test)


    positive_best_index1 = best_predict_sequence(test_result1,50,1)
    positive_best_index2 = best_predict_sequence(test_result2,50,1)

    #negative_best_index = best_predict_sequence(test_result,10,0)

    # change the activation function to linear
    layer_index1 = utils.find_layer_idx(cnn_model1, 'visualized_layer')
    layer_index2 = utils.find_layer_idx(cnn_model2, 'visualized_layer')

    cnn_model1.layers[layer_index1].activation = keras.activations.linear
    cnn_model2.layers[layer_index2].activation = keras.activations.linear


    #cnn_model.save('model_linear.h5')

    pos_grad1 = total_saliency_map(cnn_model1,dataset1,layer_index1,positive_best_index1,1)
    pos_grad2 = total_saliency_map(cnn_model2,dataset2,layer_index2,positive_best_index2,1)

    total_pos_grad1 = np.sum(pos_grad1,axis= 0)/np.sum(pos_grad1)
    total_pos_grad2 = np.sum(pos_grad2,axis= 0)/np.sum(pos_grad2)

    print(total_pos_grad1.shape)
    lineplot(total_pos_grad1,total_pos_grad2)


'''

    f, ax = plt.subplots(figsize = (20, 3))
    sns.heatmap(tf_grad, cmap='seismic',center= 0.01,ax=ax, cbar_kws={"shrink": 0.2})

    ax.set_xticks(np.arange(101))
    ax.set_xticklabels(tf_seq,rotation = 360)
    ax.set_yticks([])
    
    plt.show()
'''

if __name__ == '__main__':
    main()
