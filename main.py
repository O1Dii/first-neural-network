import keras
import numpy as np
import matplotlib.pyplot as plt

squareFeet = np.array(range(1, 9000), dtype="int16")
prices = np.array([i ** 2 / 1000000 for i in squareFeet], dtype="float16")

model = keras.Sequential()
model.add(keras.layers.Dense(20, input_dim=1, activation='relu'))
model.add(keras.layers.Dense(200, activation='elu'))
model.add(keras.layers.Dense(200, activation='elu'))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(squareFeet, prices, epochs=30, batch_size=10)

price_predict = model.predict(squareFeet)

plt.plot(squareFeet, prices, label="initial", color="red")
plt.plot(squareFeet, price_predict, label="prediction", color="blue")
plt.legend()

plt.show()
