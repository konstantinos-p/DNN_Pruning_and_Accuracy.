# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras import backend as K
K.set_image_dim_ordering('th')
import utils as ut
import numpy as np
from keras.models import load_model
#import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
path_to_model = 'cifar10/model/cifar10_model.h5'

iteration_steps = 10

#Parameters
flag1 = 1
flag2 = 1
flag3 = 1
flag4 = 1
flag5 = 1
flag6 = 1
flag7 = 1
flag8 = 1

perCsp = np.linspace(0,0.95,iteration_steps)

acc = np.zeros((iteration_steps,1))

name_of_test = 'all_layers'

#Load Model
model1 = load_model(path_to_model)

conv_1 = model1.get_layer('conv2d_1')
weights1  = conv_1.get_weights()[0]
bias1 = conv_1.get_weights()[1]

conv_2 = model1.get_layer('conv2d_2')
weights2  = conv_2.get_weights()[0]
bias2 = conv_2.get_weights()[1]

conv_3 = model1.get_layer('conv2d_3')
weights3  = conv_3.get_weights()[0]
bias3 = conv_3.get_weights()[1]

conv_4 = model1.get_layer('conv2d_4')
weights4  = conv_4.get_weights()[0]
bias4 = conv_4.get_weights()[1]

conv_5 = model1.get_layer('conv2d_5')
weights5  = conv_5.get_weights()[0]
bias5 = conv_5.get_weights()[1]

conv_6 = model1.get_layer('conv2d_6')
weights6  = conv_6.get_weights()[0]
bias6 = conv_6.get_weights()[1]

dense_1 = model1.get_layer('dense_1')
weights7  = dense_1.get_weights()[0]
bias7 = dense_1.get_weights()[1]

dense_2 = model1.get_layer('dense_2')
weights8  = dense_2.get_weights()[0]
bias8 = dense_2.get_weights()[1]

for i in range(0,iteration_steps):

    if flag1 == 1:
        weights1 = ut.compute_thresholding_sparsification(weights1, perCsp[i])
    if flag2 == 1:
        weights2 = ut.compute_thresholding_sparsification(weights2, perCsp[i])
    if flag3 == 1:
        weights3 = ut.compute_thresholding_sparsification(weights3, perCsp[i])
    if flag4 == 1:
        weights4 = ut.compute_thresholding_sparsification(weights4, perCsp[i])
    if flag5 == 1:
        weights5 = ut.compute_thresholding_sparsification(weights5, perCsp[i])
    if flag6 == 1:
        weights6 = ut.compute_thresholding_sparsification(weights6, perCsp[i])
    if flag7 == 1:
        weights7 = ut.compute_thresholding_sparsification(weights7, perCsp[i])
    if flag8 == 1:
        weights8 = ut.compute_thresholding_sparsification(weights8, perCsp[i])

    acc[i] =  ut.evaluate_cifar_keras(path_to_model,X_test,y_test,[weights1,bias1],[weights2,bias2],[weights3,bias3],[weights4,bias4],[weights5,bias5],[weights6,bias6],[weights7,bias7],[weights8,bias8])[1]
    print('Calculating Step: ', i  )

#plt.plot(perCsp*100, acc*100)

np.save('cifar10/results/'+name_of_test+'_sp.npy',perCsp*100)
np.save('cifar10/results/'+name_of_test+'_ac.npy',acc*100)

end = 1