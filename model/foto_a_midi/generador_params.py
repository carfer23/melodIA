import numpy as np
import glob
import os

# Prefijos y función generadora asociada
prefijos_generadores = {
    "bright_sky": lambda: [
        np.random.uniform(0.6, 1.0), np.random.randint(100, 160), np.random.uniform(0.5, 1.5),
        np.random.uniform(0.2, 0.5), np.random.uniform(0.6, 1.0), np.random.uniform(0.4, 0.7),
        np.random.uniform(0.6, 1.0), 1, np.random.uniform(0.4, 0.7), np.random.randint(2, 4)
    ],
    "cloudy_sky": lambda: [
        np.random.uniform(0.3, 0.7), np.random.randint(80, 120), np.random.uniform(0.6, 2.0),
        np.random.uniform(0.3, 0.6), np.random.uniform(0.5, 0.8), np.random.uniform(0.3, 0.6),
        np.random.uniform(0.4, 0.8), 1, np.random.uniform(0.3, 0.6), np.random.randint(2, 4)
    ],
    "rainbow": lambda: [
        np.random.uniform(0.7, 1.0), np.random.randint(100, 160), np.random.uniform(0.4, 1.2),
        np.random.uniform(0.1, 0.4), np.random.uniform(0.7, 1.0), np.random.uniform(0.4, 0.7),
        np.random.uniform(0.7, 1.0), 1, np.random.uniform(0.5, 0.8), np.random.randint(2, 4)
    ],
    "rainy_sky": lambda: [
        np.random.uniform(0.1, 0.4), np.random.randint(60, 100), np.random.uniform(1.0, 2.5),
        np.random.uniform(0.5, 0.9), np.random.uniform(0.5, 0.7), np.random.uniform(0.2, 0.5),
        np.random.uniform(0.2, 0.6), 0, 0.0, np.random.randint(1, 3)
    ],
    "starry_night": lambda: [
        np.random.uniform(0.2, 0.5), np.random.randint(40, 100), np.random.uniform(1.5, 3.0),
        np.random.uniform(0.6, 0.9), np.random.uniform(0.5, 0.7), np.random.uniform(0.2, 0.5),
        np.random.uniform(0.5, 0.9), 1, np.random.uniform(0.3, 0.6), np.random.randint(2, 4)
    ],
    "stormy_sky": lambda: [
        np.random.uniform(0.0, 0.3), np.random.randint(60, 120), np.random.uniform(1.0, 2.5),
        np.random.uniform(0.5, 0.9), np.random.uniform(0.5, 0.8), np.random.uniform(0.6, 1.0),
        np.random.uniform(0.3, 0.6), 0, 0.0, np.random.randint(1, 3)
    ],
    "sunset_with_clouds": lambda: [
        np.random.uniform(0.5, 0.8), np.random.randint(80, 140), np.random.uniform(0.6, 2.0),
        np.random.uniform(0.2, 0.6), np.random.uniform(0.6, 1.0), np.random.uniform(0.3, 0.6),
        np.random.uniform(0.6, 1.0), 1, np.random.uniform(0.3, 0.6), np.random.randint(2, 4)
    ],
    "thunderstorm": lambda: [
        np.random.uniform(0.0, 0.2), np.random.randint(80, 140), np.random.uniform(0.8, 2.0),
        np.random.uniform(0.7, 1.0), np.random.uniform(0.5, 0.9), np.random.uniform(0.6, 1.0),
        np.random.uniform(0.2, 0.5), 0, 0.0, np.random.randint(1, 3)
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