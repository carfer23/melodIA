import numpy as np
import torch
import clip
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .modelo import construir_modelo, clip_parametros
from .utils import cargar_imagen

class FotoAMIDI:
    def __init__(self, modelo_ruta='modelo_regresion_midi.h5', usar_clip=True, clip_model_name="ViT-B/32"):
        self.usar_clip = usar_clip
        self.clip_model_name = clip_model_name
        self.modelo_ruta = modelo_ruta
        self.modelo = None
        self.scaler = None
        self.nombres_parametros = [
            "tonalidad_value", "tempo", "duracion_media", "sigma",
            "velocidad_media", "densidad_media", "caracter_melodico",
            "usar_acordes", "proporcion_acordes", "rango_octavas"
        ]

        if usar_clip:
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name)
            self.clip_model = self.clip_model.to("cuda" if torch.cuda.is_available() else "cpu")
            self.input_size = self.clip_model.visual.input_resolution
        else:
            self.input_size = (224, 224)

    def preprocesar_imagen(self, imagen):
        if self.usar_clip:
            return self.clip_preprocess(imagen).unsqueeze(0)
        else:
            imagen = imagen.resize(self.input_size)
            arr = np.array(imagen) / 255.0
            return np.expand_dims(arr, axis=0)

    def predict(self, imagen):
        if self.modelo is None:
            self.load_modelo()

        entrada = self.preprocesar_imagen(imagen)

        if self.usar_clip:
            with torch.no_grad():
                tensor = entrada.to("cuda" if torch.cuda.is_available() else "cpu")
                features = self.clip_model.encode_image(tensor)
                salida = self.modelo.predict(features.cpu().numpy())[0]
        else:
            salida = self.modelo.predict(entrada)[0]

        salida = clip_parametros(salida)
        return dict(zip(self.nombres_parametros, salida))

    def load_modelo(self):
        self.modelo = keras.models.load_model(self.modelo_ruta)

    def train(self, rutas_imagenes, parametros_midi, epochs=100, batch_size=32, validation_split=0.2):
        imagenes = []
        for ruta in rutas_imagenes:
            img = cargar_imagen(ruta)
            proc = self.preprocesar_imagen(img)
            imagenes.append(proc)

        if self.usar_clip:
            imagenes_tensor = torch.cat(imagenes, dim=0).to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                X = self.clip_model.encode_image(imagenes_tensor).cpu().numpy()
        else:
            X = np.concatenate(imagenes, axis=0)

        y = np.array(parametros_midi)
        self.scaler = StandardScaler()
        y = self.scaler.fit_transform(y)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)

        if self.modelo is None:
            self.modelo = construir_modelo(input_shape=X_train.shape[1:], output_size=len(self.nombres_parametros))

        self.modelo.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        self.modelo.save(self.modelo_ruta)
