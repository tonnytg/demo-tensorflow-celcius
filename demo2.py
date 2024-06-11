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
predicted_fahrenheit = model.predict(celsius_q)

# Plotando os dados originais e as previsões
plt.figure(figsize=(10, 6))
plt.plot(celsius_q, fahrenheit_a, 'ro', label='Dados Reais')
plt.plot(celsius_q, predicted_fahrenheit, 'b-', label='Previsões do Modelo')
plt.xlabel('Celsius')
plt.ylabel('Fahrenheit')
plt.title('Conversão de Celsius para Fahrenheit')
plt.legend()
plt.grid(True)
plt.show()

# Plotando a perda ao longo das épocas
plt.figure(figsize=(10, 6))
plt.xlabel('Número de Épocas')
plt.ylabel('Magnitude da Perda')
plt.plot(history.history['loss'])
plt.title('Perda durante o Treinamento')
plt.grid(True)
plt.show()
