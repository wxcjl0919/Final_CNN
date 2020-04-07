
from tensorflow.keras.models import load_model
from vis.visualization import visualize_saliency
from vis.utils import utils
import numpy as np
from cnn.generate_data import data_loader
import keras
import seaborn as sns
import matplotlib.pyplot as plt

dataset_name = 'K562_UUP_UUE'
date ='3_25_2020'


datasets = data_loader('/Users/fais/Desktop/researches/results/2_26_2020/dataset/' + dataset_name +'.h5')
cnn_model = load_model('/Users/fais/Desktop/researches/results/'+date+'/' + dataset_name + '/best_model.h5')

pred_result = cnn_model(datasets.X_test)

promoter_predict_result = np.round(pred_result,decimals=3)[:,1]
best_promoter_result_index = np.argsort(-promoter_predict_result)[0:30]

enhancer_predict_result = np.round(pred_result,decimals=3)[:,0]
best_enhancer_result_index = np.argsort(-enhancer_predict_result)[0:30]


layer_index = utils.find_layer_idx(cnn_model, 'visualized_layer')
cnn_model.layers[layer_index].activation = keras.activations.linear

cnn_model.save('model_linear.h5')
cnn_model = load_model('model_linear.h5')






def saliency_map(index,filter_indices):

    total_grad = np.zeros(500)
    for x in range(len(index)):

        _,grads = visualize_saliency(model= cnn_model,
                                     layer_idx= layer_index,
                                     filter_indices = filter_indices,
                                     seed_input= datasets.X_test[index[x]])



        total_grad = grads + total_grad

    total_grad = np.round(total_grad, decimals=3)

    return total_grad




promoter_saliency_map = saliency_map(best_promoter_result_index,1)
enhancer_saliency_map = saliency_map(best_enhancer_result_index,0)

promoter_saliency_map = promoter_saliency_map/ np.sum(promoter_saliency_map)
enhancer_saliency_map = enhancer_saliency_map / np.sum(enhancer_saliency_map)

total_saliency_map = np.vstack((promoter_saliency_map,enhancer_saliency_map))

x = np.arange(0,500)
random = np.ones(500)/500
def heatmap(saliency_map):

     sns.heatmap(total_saliency_map, cmap='seismic', yticklabels=['promoter', 'enhancer'], xticklabels=50, center=0.002)

     plt.show()



heatmap(total_saliency_map)

def lineplot(x,promoter,enhancer):

    random = np.ones(500) / 500
    zero = np.zeros(500)

    plt.plot(x,promoter[0],color = 'red',label = 'promoter')
    plt.plot(x, enhancer[0], color='blue', label='enhancer')

    promoter_enhancer = promoter[0] - enhancer[0]
    plt.plot(x,promoter_enhancer,color = 'green',label = 'promoter - enhancer')
    plt.plot(x,random,color = 'grey',label = 'random')
    plt.plot(x,zero,color = 'grey', label = '0')
    plt.yticks(np.arange(-0.003,0.006,0.001))
    plt.xticks(np.arange(0,550,50))
    plt.xlabel('positions')
    plt.ylabel('importance')

    plt.legend()
    plt.show()


#lineplot(x,promoter_saliency_map,enhancer_saliency_map)



