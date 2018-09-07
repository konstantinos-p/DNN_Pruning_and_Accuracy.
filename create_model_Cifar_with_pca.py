# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
from sklearn.decomposition import PCA
import utils as ut

# fix random seed for reproducibility
#seed = 7
#numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Parameters
pca_copmponents = 350
iteration_steps = 50
outer_instances = 10

#Create Augmentation
X_train = np.mean(X_train,axis = 1)
X_test = np.mean(X_test,axis = 1)
Χ_train_unfold = np.reshape(X_train,(X_train.shape[0],-1))
Χ_test_unfold = np.reshape(X_test,(X_test.shape[0],-1))

pca  = PCA(n_components=pca_copmponents)
pca.fit(Χ_test_unfold)

X_train  = pca.transform(Χ_train_unfold)
X_test  = pca.transform(Χ_test_unfold)

X_train  = pca.inverse_transform(X_train)
X_test  = pca.inverse_transform(X_test)

X_train = np.reshape(X_train,(X_train.shape[0],1,32,32))
X_test = np.reshape(X_test,(X_test.shape[0],1,32,32))

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(1,32, 32), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))


# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model


# Evaluate first layer for different sparsity levels
name_of_file = 'pca_layer0_ncom'+str(pca_copmponents)
perCsp = np.linspace(0,0.95,iteration_steps)
acc = np.zeros((iteration_steps,outer_instances))

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


for j in range(0,outer_instances):


    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)

    conv_original = model.get_layer('conv2d_1')
    weights_original  = conv_original.get_weights()[0]
    bias_original = conv_original.get_weights()[1]



    for i in range(0,iteration_steps):
        weights_tmp = ut.compute_thresholding_sparsification(weights_original, perCsp[i])
        conv_original.set_weights([weights_tmp,bias_original])
        acc[i,j] = model.evaluate(X_test, y_test, verbose=0)[1]

    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    reset_weights(model)

np.save('cifar_10_with_pca/results/'+name_of_file+'_mean_acc.npy',np.mean(acc,axis=1))
np.save('cifar_10_with_pca/results/'+name_of_file+'_var_acc.npy',np.var(acc,axis=1))
np.save('cifar_10_with_pca/results/'+name_of_file+'_sp.npy',perCsp)

end  = 1