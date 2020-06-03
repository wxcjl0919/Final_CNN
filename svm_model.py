from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import numpy as np
import h5py


def load_data(data_path):

    dataset = h5py.File(data_path, 'r')

    X_train = np.array(dataset['X_train']).astype(np.float32)
    Y_train = np.array(dataset['Y_train']).astype(np.float32)
    X_valid = np.array(dataset['X_test']).astype(np.float32)
    Y_valid = np.array(dataset['Y_test']).astype(np.float32)

    return X_train,Y_train,X_valid,Y_valid


def svm_model(X_train,Y_train,X_valid,Y_valid):

    svc = svm.SVC(kernel= 'rbf', class_weight= 'balanced')
    c_range = np.logspace(-3, 1, 5, base=2)
    gamma_range = np.logspace(-13, -1, 5, base=2)

    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)

    clf =grid.fit(X_train,Y_train)

    print(clf.best_params_)

    f1_scores = metrics.f1_score(Y_valid, clf.predict(X_valid))
    fpr, tpr, thresholds = metrics.roc_curve(Y_valid, clf.decision_function(X_valid),pos_label = 1)

    auc_scores = metrics.auc(fpr,tpr)


    return f1_scores, auc_scores



def main():

    dataset_name = 'K562_US_UU.h5'
    dataset_path = '/Users/xinzeng/Desktop/research/result/5_27_2020/'

    X_train,Y_train,X_valid,Y_valid = load_data(dataset_path + dataset_name)


    Y_train = Y_train.flatten()
    Y_valid = Y_valid.flatten()


    f1_scores,auc_scores = svm_model(X_train,Y_train,X_valid,Y_valid)

    print(f1_scores)
    print(auc_scores)



if __name__ == '__main__':
    main()
