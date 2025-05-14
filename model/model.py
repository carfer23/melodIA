# model.py

import pandas as pd
import numpy as np
import pretty_midi
import collections
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
from IPython import display

MAESTRO_PATH = "../maestro-v3.0.0/"
MODEL_PATH = 'melodIAmodel.keras'

COLUMNS = ["pitch", "step", "duration", "velocity"]

RANDOM_STATE = 13
_SAMPLING_RATE = 16000
SEQUENCE_LENGTH = 128

VELOCITY_MIN = 1
VELOCITY_MAX = 127
PITCH_MIN = 0
PITCH_MAX = 127


class MelodIAModel:
    """
    Engloba las funciones necesarias para trabajar con el modelo.
    """

    def load_model():
        """
        Carga el modelo.
        """
        tf.keras.utils.get_custom_objects().update({'mse_with_positive_pressure': MelodIAModel.mse_with_positive_pressure})
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Modelo cargado desde: {MODEL_PATH}")
            model.summary()
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
        return model

    def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Función de coste (loss function) personalizada. Se basa en el MSE (error cuadrático medio) 
        añadiendo una penalización a las predicciones negativas. De esta forma hacemos que el modelo 
        "tienda" a generar valores no negativos (para pitch, duration y velocity).
        """
        mse = (y_true - y_pred) ** 2
        positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
        return tf.reduce_mean(mse + positive_pressure)

    def normalize(notes: np.ndarray, normalize_pitch=True, scale_velocity=True):
        """
        Normaliza el pitch y la velocity en un array de NumPy.
        """
        if normalize_pitch:
            pitch_min = 0
            pitch_max = 127
            notes[:, 0] = (notes[:, 0] - pitch_min) / (pitch_max - pitch_min)
        
        if scale_velocity:
            velocity_min = 1
            velocity_max = 127
            notes[:, 3] = (notes[:, 3] - velocity_min) / (velocity_max - velocity_min)
        
        return notes

    def desnormalize(generated_notes: pd.DataFrame):
        """
        Desnormaliza el pitch y la velocity en un DataFrame.
        """
        generated_notes['pitch'] = (generated_notes['pitch'] * (PITCH_MAX - PITCH_MIN) + PITCH_MIN).astype(int)
        generated_notes['velocity'] = (generated_notes["velocity"] * (VELOCITY_MAX - VELOCITY_MIN) + VELOCITY_MIN).astype(int)
        return generated_notes
    
    def predict_next_note(notes: np.ndarray, keras_model: tf.keras.Model, temperature: float = 1.0) -> int:
        """
        Genera la nota inmediatamente siguiente a la secuencia de notas pasada como input.
        """
        assert temperature > 0

        # Add batch dimension
        inputs = tf.expand_dims(notes, 0)

        predictions = keras_model.predict(inputs)
                                            
        pitch_logits = predictions['pitch']
        step = predictions['step']
        duration = predictions['duration']
        velocity = predictions['velocity']

        pitch_logits /= temperature
        pitch = tf.random.categorical(pitch_logits, num_samples=1)
        pitch = tf.squeeze(pitch, axis=-1)
        duration = tf.squeeze(duration, axis=-1)
        step = tf.squeeze(step, axis=-1)
        velocity = tf.squeeze(velocity, axis=-1)

        # step, duration y velocity no deben ser negativos
        step = tf.maximum(0, step)
        duration = tf.maximum(0, duration)
        velocity = tf.maximum(0, velocity)

        return int(pitch), float(step), float(duration), float(velocity)
    
    def predict_sequence(keras_model: tf.keras.Model, input_notes: np.ndarray, num_predictions: int = 50, temperature: float = 1.0) -> pd.DataFrame:
        """
        Genera una secuencia de notas predichas por el modelo, continuando a partir de una secuencia inicial.
        IMPORTANTE: los valores de pitch y velocity se devuelven normalizados.
        """
        generated_notes = []
        prev_start = 0

        for _ in range(num_predictions):
            # Predice la nota inmesiatamente siguiente a la secuencia
            pitch, step, duration, velocity = MelodIAModel.predict_next_note(input_notes, keras_model, temperature)

            # Normalizamos el pitch ya que lo volveremos a introducir al modelo para la siguiente predicción
            normalized_pitch = (pitch - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)

            start = prev_start + step
            end = start + duration
            input_note = (normalized_pitch, step, duration, velocity)
            generated_notes.append((*input_note, start, end))

            # Eliminamos la primera nota de la secuencia y añadimos la nota predicha como última nota
            input_notes = np.delete(input_notes, 0, axis=0)
            input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
            prev_start = start

        generated_df = pd.DataFrame(
            generated_notes, columns=(*COLUMNS, 'start', 'end'))
        return generated_df


class Utilities:
    """
    Esta clase contiene varias funciones para manejar tanto los objetos PrettyMidi como el resto de
    estructuras de datos que usamos para entrenar y utilizar el modelo.
    """

    @staticmethod
    def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
        """
        Muestra un piano roll con todas las notas de un objeto PrettyMIDI.
        """
        plt.figure(figsize=(20, 4))
        librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                                hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                                fmin=pretty_midi.note_number_to_hz(start_pitch))

    @staticmethod 
    def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
        """
        Genera un reproductor de audio que se puede usar directamente en un notebook.
        """
        waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
        # Take a sample of the generated waveform to mitigate kernel resets
        waveform_short = waveform[:seconds*_SAMPLING_RATE]
        return display.Audio(waveform_short, rate=_SAMPLING_RATE)
    
    @staticmethod
    def get_sample_midi_path(df):
        """
        Obtiene una ruta a un archivo midi aleatorio.
        """
        sample = df.sample(n=1, random_state=RANDOM_STATE, ignore_index=True)
        sample_file_path = "../maestro-v3.0.0/" + sample.loc[0]["midi_filename"]

        composer = sample.loc[0]["canonical_composer"]
        title = sample.loc[0]["canonical_title"]
        print(f"{composer} - {title}")

        return sample_file_path
    
    @staticmethod
    def midi_path_to_df(midi_file: str) -> pd.DataFrame:
        """
        Lee el archivo midi de la ruta especificada y extrae la información
        de sus notas a un DataFrame.
        """
        pm = pretty_midi.PrettyMIDI(midi_file)
        instrument = pm.instruments[0]
        notes = collections.defaultdict(list)

        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['name'].append(pretty_midi.note_number_to_name(note.pitch))
            notes['pitch'].append(note.pitch)
            notes['start'].append(start)
            notes['end'].append(end)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            notes['velocity'].append(note.velocity)
            prev_start = start

        return pd.DataFrame({name: np.array(value) for name, value in notes.items()})
    
    @staticmethod 
    def midi_to_numpy(mid: pretty_midi.PrettyMIDI, num_notes=25) -> np.array:
        """
        Parte de un objeto PrettyMIDI y extrae las características de
        las notas en un array de NumPy.
        """

        notes = collections.defaultdict(list) # diccionario (si no existe una clave, la crea)
        raw_notes = mid.instruments[0].notes # notas del archivo midi
        prev_start = raw_notes[0].start # start de la nota anterior

        for note in raw_notes:
            # Características de la nota
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            notes['velocity'].append(note.velocity)

            prev_start = start

        notes = np.stack([notes[key] for key in ['pitch', 'step', 'duration', 'velocity']], axis=-1)
        return notes
    
    @staticmethod 
    def notes_to_midi(notes: pd.DataFrame, instrument_name='Acoustic Grand Piano') -> pretty_midi.PrettyMIDI:
        """
        Transforma un DataFrame de notas a un objeto de PrettyMIDI.
        """

        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(
            program=pretty_midi.instrument_name_to_program(
                instrument_name))

        prev_start = 0
        for i, note in notes.iterrows():
            start = float(prev_start + note['step'])
            end = float(start + note['duration'])
            velocity = int(note['velocity'])
            pitch = int(note['pitch'])

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=end,
            )
            instrument.notes.append(note)
            prev_start = start

        pm.instruments.append(instrument)
        
        return pm
