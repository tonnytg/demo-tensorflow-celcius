import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Dados de entrada (Celsius)
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
# Dados de saída (Fahrenheit)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Definindo o modelo com uma camada densa
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compilando o modelo com otimizador Adam e perda de erro quadrático médio
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Treinando o modelo
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=0)

# Fazendo previsões
predicted_fahrenheit = model.predict(celsius_q)

# Função para plotar os dados reais e as previsões do modelo
def plot_predictions(celsius, real_fahrenheit, predicted_fahrenheit):
    plt.figure(figsize=(10, 6))
    plt.plot(celsius, real_fahrenheit, 'ro', label='Dados Reais')
    plt.plot(celsius, predicted_fahrenheit, 'b-', label='Previsões do Modelo')
    plt.xlabel('Celsius')
    plt.ylabel('Fahrenheit')
    plt.title('Conversão de Celsius para Fahrenheit')
    plt.legend()
    plt.grid(True)
    plt.show()

# Função para plotar a perda ao longo das épocas
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.xlabel('Número de Épocas')
    plt.ylabel('Magnitude da Perda')
    plt.title('Perda durante o Treinamento')
    plt.grid(True)
    plt.show()

# Chamando as funções de plotagem
plot_predictions(celsius_q, fahrenheit_a, predicted_fahrenheit)
plot_loss(history)
