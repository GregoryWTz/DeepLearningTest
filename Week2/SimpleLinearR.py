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
# plt.scatter(data['area'], data['price'])
# plt.xlabel('Area')
# plt.ylabel('Price')
# plt.title('Example Data')
# plt.show()

#Calculating the two regression coefficients using the equations
W = sum(price*(area - np.mean(area))) / sum((area - np.mean(area))**2)
b = np.mean(price) - W * np.mean(area)
print("======================================")
print("The calculated regression coefficients are: ", W, b)
print("======================================\n")

#predicting the price using the calculated coefficients
y_predicted_price = W * area + b

#Plotting the predicted price against the actual price
plt.plot(area, y_predicted_price, color = 'red', label = 'Predicted Price')
plt.scatter(data['area'], data['price'], label = 'Actual Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.title('Predicted vs Actual Price')
plt.show()