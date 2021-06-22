from minibrain import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd


x = np.array([[random.uniform(0, 300) for _ in range(2)] for _ in range(1000)])
x = preprocessing.normalize(x)
y = np.array([[i[0] - i[1]] for i in x])


trainPorc = 95
trainSize = int(len(y) * trainPorc / 100)

x_train = x[:trainSize]
y_train = y[:trainSize]

x_test = x[trainSize:]
y_test = y[trainSize:]

model = NeuralNetwork(2, [20], 1, activaction_function="relu")

model.train(x_train, y_train, 100, 0.1)



input = np.array(x_test)


output = model.predict(input)


print(model.predict([150, 80]))

fig = plt.figure( figsize=(10, 6), dpi=80)
plt.plot(y_test, 'b', label="Teste")
plt.plot(output, 'r', label="Hora do vamo ve")
plt.legend()
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
