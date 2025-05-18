import numpy as np
import glob
import os

class MidiParamsGenerator:
    """
    Genera parámetros MIDI y rutas de imágenes basados en prefijos
    y funciones generadoras proporcionados por el usuario.
    """

    def __init__(self, prefijos_generadores, image_folder="./datos/pexels_cielos_varios"):
        self.prefijos_generadores = prefijos_generadores
        self.image_folder = image_folder
        self.parametros_midi = []
        self.rutas_imagenes = []

    def generar_parametros_y_rutas(self):
        """Busca imágenes por prefijo y genera parámetros MIDI asociados."""
        for prefijo, generador in self.prefijos_generadores.items():
            patrones = os.path.join(self.image_folder, f"{prefijo}*")
            for ruta in glob.glob(patrones):
                self.rutas_imagenes.append(ruta)
                self.parametros_midi.append(generador())

    def guardar_parametros(self, filename="parametros_midi.npy"):
        """Guarda los parámetros MIDI generados en un archivo .npy."""
        np.save(filename, np.array(self.parametros_midi))

    def guardar_rutas(self, filename="rutas_imagenes.txt"):
        """Guarda las rutas de las imágenes en un archivo de texto."""
        with open(filename, "w") as f:
            for ruta in self.rutas_imagenes:
                f.write(ruta + "\n")
