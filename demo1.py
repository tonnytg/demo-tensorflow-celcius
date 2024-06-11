import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Dados de entrada (Celsius)
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
# Dados de saída (Fahrenheit)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Definindo o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compilando o modelo
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Treinando o modelo
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)

# Fazendo previsões
print(model.predict([100.0]))

# Visualizando a perda ao longo das épocas
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])
plt.show()
