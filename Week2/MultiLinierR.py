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

plt.show()