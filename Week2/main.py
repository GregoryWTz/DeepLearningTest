import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Generating a random data
np.random.seed(0)
area = 2.5 * np.random.randn(100) + 25
price = 25 * area + 5 + np.random.randint(20,50, size = len(area))
data = np.array([area, price])
data = pd.DataFrame(data.T, columns = ['area', 'price'])
plt.scatter(data['area'], data['price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Example Data')
plt.show()