# Week 1 - Deep Learning
# Dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

import numpy as np
import idx2numpy
import tensorflow as tf
from tensorflow import keras

# Load the MNIST dataset
X_train = idx2numpy.convert_from_file('Dataset/train-images.idx3-ubyte')
Y_train = idx2numpy.convert_from_file('Dataset/train-labels.idx1-ubyte')
X_test = idx2numpy.convert_from_file('Dataset/t10k-images.idx3-ubyte')
Y_test = idx2numpy.convert_from_file('Dataset/t10k-labels.idx1-ubyte')

# Preprocess the data
NB_CLASSES = 10
RESHAPED = 784

# Reshape: flatten 28x28 grid into a 784-length vector
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)

# Cast to float32 so we can do decimal math during training
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize pixel values from [0,255] to [0,1]
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# network and training parameters
EPOCHS = 50
BATCH_SIZE = 128
VERBOSE = 1
VALIDATION_SPLIT = 0.2
N_HIDDEN = 128
DROPOUT = 0.3

model = tf.keras.models.Sequential()

model.add(keras.layers.Dense(N_HIDDEN,
    input_shape=(RESHAPED,),
    name='dense_layer1',
    activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))

model.add(keras.layers.Dense(N_HIDDEN,
    name='dense_layer2',
    activation='relu'))
model.add(keras.layers.Dropout(DROPOUT))

model.add(keras.layers.Dense(NB_CLASSES,
    name='dense_layer3',
    activation='relu'))

for i in range(16):
    model.add(keras.layers.Dense(N_HIDDEN,
        name=f'dense_layer{i+4}',
        activation='relu'))

model.add(keras.layers.Dense(NB_CLASSES,
    name='dense_layer180',
    activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=VERBOSE,
          validation_split=VALIDATION_SPLIT)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)