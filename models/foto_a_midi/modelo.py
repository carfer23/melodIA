import numpy as np
import tensorflow as tf
from tensorflow import keras

def construir_modelo(input_shape, output_size):
    modelo = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=input_shape),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(output_size)
    ])
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

def clip_parametros(parametros):
    parametros[0] = np.clip(parametros[0], 0, 1)
    parametros[1] = int(np.clip(parametros[1], 20, 200))
    parametros[2] = np.clip(parametros[2], 0.1, 4)
    parametros[3] = np.clip(parametros[3], 0.1, 1)
    parametros[4] = np.clip(parametros[4], 0, 1)
    parametros[5] = np.clip(parametros[5], 0, 1)
    parametros[6] = np.clip(parametros[6], 0, 1)
    parametros[7] = int(np.clip(parametros[7], 0, 1) > 0.5)
    parametros[8] = np.clip(parametros[8], 0, 1)
    parametros[9] = int(np.clip(parametros[9], 1, 4))
    return parametros
