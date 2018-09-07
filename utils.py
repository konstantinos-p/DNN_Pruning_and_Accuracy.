from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
from keras.models import Model

def evaluate_cifar_keras(path_to_model,X_test,y_test,new_dense_1,new_dense_2,new_dense_3,new_dense_4,new_dense_5,new_dense_6,new_dense_7,new_dense_8):

    #load data and preprocess
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # normalize inputs from 0-255 to 0.0-1.0
    X_test = X_test.astype('float32')
    X_test = X_test / 255.0
    # one hot encode outputs
    y_test = np_utils.to_categorical(y_test)

    # Load model and Test
    model = load_model(path_to_model)


    #Set Layers


    dense_1 = model.get_layer('conv2d_1')
    dense_1.set_weights(new_dense_1)

    dense_2 = model.get_layer('conv2d_2')
    dense_2.set_weights(new_dense_2)

    dense_3 = model.get_layer('conv2d_3')
    dense_3.set_weights(new_dense_3)

    dense_4 = model.get_layer('conv2d_4')
    dense_4.set_weights(new_dense_4)

    dense_5 = model.get_layer('conv2d_5')
    dense_5.set_weights(new_dense_5)

    dense_6 = model.get_layer('conv2d_6')
    dense_6.set_weights(new_dense_6)

    dense_7 = model.get_layer('dense_1')
    dense_7.set_weights(new_dense_7)

    dense_8 = model.get_layer('dense_2')
    dense_8.set_weights(new_dense_8)


    scores = model.evaluate(X_test, y_test, verbose=0)
    return scores

def compute_thresholding_sparsification(W,perCsp):
    #takes as input a matrix thresholds it based on magnitude to required sparsity


    b = np.reshape(np.abs(W), (-1))
    hist, bin_edges = np.histogram(b, bins=100, density=True)
    hist = hist / np.sum(hist)
    cumulative = np.cumsum(hist)
    pos = np.where(cumulative >= perCsp)
    threshold = bin_edges[pos[0][0]]


    W[np.where(np.abs(W) < threshold)] = 0

    # Calculate Sparsity in sanity check
    pos2 = np.where(W == 0)

    total_elements = 1
    for i in range(0,len(W.shape)):
        total_elements = W.shape[i]*total_elements

    sparse = pos2[0].shape[0] / (total_elements)
    print("Sparsified W to: ",sparse*100," sparsity")

    return W