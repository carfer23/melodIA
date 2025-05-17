import numpy as np
import glob
import os

# Prefijos y función generadora asociada
prefijos_generadores = {
    "thunderstorm": lambda: [
        np.random.uniform(0.0, 0.1), np.random.randint(130, 160), np.random.uniform(2.0, 2.5),
        np.random.uniform(0.8, 1.0), np.random.uniform(0.7, 1.0), np.random.uniform(0.9, 1.0),
        np.random.uniform(0.5, 0.7), 0, 0.0, np.random.randint(1, 2)
    ],
    "rainy_sky": lambda: [
        np.random.uniform(0.1, 0.2), np.random.randint(90, 110), np.random.uniform(1.8, 2.2),
        np.random.uniform(0.6, 0.8), np.random.uniform(0.5, 0.7), np.random.uniform(0.5, 0.6),
        np.random.uniform(0.5, 0.7), 0, 0.0, np.random.randint(1, 2)
    ],
    "bright_sunny_sky": lambda: [
        np.random.uniform(0.9, 1.0), np.random.randint(140, 160), np.random.uniform(0.4, 0.7),
        np.random.uniform(0.1, 0.2), np.random.uniform(0.9, 1.0), np.random.uniform(0.6, 0.8),
        np.random.uniform(0.9, 1.0), 1, np.random.uniform(0.6, 0.8), np.random.randint(3, 4)
    ],
    "rainbow": lambda: [
        np.random.uniform(0.8, 1.0), np.random.randint(120, 140), np.random.uniform(0.5, 0.8),
        np.random.uniform(0.2, 0.3), np.random.uniform(0.8, 1.0), np.random.uniform(0.6, 0.8),
        np.random.uniform(0.9, 1.0), 1, np.random.uniform(0.7, 0.8), np.random.randint(3, 4)
    ],
    "starry_night": lambda: [
        np.random.uniform(0.2, 0.3), np.random.randint(60, 80), np.random.uniform(2.5, 3.0),
        np.random.uniform(0.8, 1.0), np.random.uniform(0.6, 0.7), np.random.uniform(0.4, 0.5),
        np.random.uniform(0.5, 0.7), 1, np.random.uniform(0.5, 0.6), np.random.randint(2, 3)
    ],
    "clouds": lambda: [
        np.random.uniform(0.4, 0.5), np.random.randint(90, 110), np.random.uniform(1.4, 1.6),
        np.random.uniform(0.5, 0.6), np.random.uniform(0.6, 0.7), np.random.uniform(0.5, 0.6),
        np.random.uniform(0.6, 0.7), 1, np.random.uniform(0.5, 0.6), np.random.randint(2, 3)
    ],
    "sunset": lambda: [
        np.random.uniform(0.7, 0.9), np.random.randint(110, 130), np.random.uniform(1.5, 2.0),
        np.random.uniform(0.4, 0.5), np.random.uniform(0.7, 0.9), np.random.uniform(0.5, 0.6),
        np.random.uniform(0.7, 0.9), 1, np.random.uniform(0.6, 0.7), np.random.randint(2, 3)
    ],
    "snow": lambda: [
        np.random.uniform(0.2, 0.3), np.random.randint(70, 90), np.random.uniform(1.0, 1.3),
        np.random.uniform(0.4, 0.5), np.random.uniform(0.8, 1.0), np.random.uniform(0.6, 0.7),
        np.random.uniform(0.6, 0.8), 1, np.random.uniform(0.5, 0.6), np.random.randint(2, 3)
    ]
}




# Ruta donde están las imágenes
ruta_imagenes = "./datos/pexels_cielos_varios"
parametros_midi = []
rutas_imagenes = []

for prefijo, generador in prefijos_generadores.items():
    for ruta in glob.glob(os.path.join(ruta_imagenes, f"{prefijo}*")):
        rutas_imagenes.append(ruta)
        parametros_midi.append(generador())

# Guardar resultados
np.save("parametros_midi.npy", np.array(parametros_midi))
with open("rutas_imagenes.txt", "w") as f:
    f.writelines([ruta + "\n" for ruta in rutas_imagenes])