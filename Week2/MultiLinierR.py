# Multiple and Multivariate Linear Regression using TensorFlow Keras APi
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras as K
import seaborn as sns
from tensorflow.keras.layers import Dense, Normalization

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']

data = pd.read_csv(url, names=columns, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

data = data.drop('origin', axis=1)
print(data.isna().sum())
data = data.dropna()

train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)

sns.pairplot(train_dataset[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']], diag_kind='kde')

# plt.show()
plt.close()

# seperate variables into input variables and the label that we want to predict
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('mpg')
test_labels = test_features.pop('mpg')

# Normalize the data
data_normalizer = Normalization(axis = -1)
data_normalizer.adapt(np.array(train_features))

model = K.Sequential([
    data_normalizer,
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation=None)
])

model.summary()

# Compile the model with Adam optimizer and mean squared error loss function
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])

# Train the model for 100 epochs with 20% of the training data used for validation
history = model.fit(x=train_features, y=train_labels, epochs=100, validation_split=0.2, verbose=1)

plt.plot(history.history['loss'], label='loss', color='black', linewidth=1.5)
plt.plot(history.history['val_loss'], label='val_loss', color='red', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('Error [MPG]')
plt.legend()
plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.7)
plt.show()