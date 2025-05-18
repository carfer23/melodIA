import pretty_midi
import numpy as np
import random

class ParamsAMIDI:
    """
    Genera archivos MIDI a partir de parámetros que describen características musicales.
    """
    def __init__(self, tonalidad_value=0.8, tempo=120, duracion_media=1, sigma=0.5, velocidad_media=0.7, densidad_media=1,caracter_melodico=0.8,usar_acordes=1, proporcion_acordes=0.5, rango_octavas=1): # Añadidos parámetros
        """
        Inicializa la clase con los parámetros de tonalidad, tempo, y duración del MIDI.

        Args:
            tonalidad_value (float): Un valor entre 0 y 1 que representa la tonalidad.
            tempo (int): Tempo en BPM (por defecto 120).
            duracion_media (float): Duración media de las notas.
            sigma (float): Desviación estándar de la distribución de duraciones.
            velocidad_media (float): Velocidad media de las notas (0 a 1).
            densidad_media (float): Densidad de las notas (0 a 1).
            caracter_melodico (float): Carácter melódico (0 a 1).
            usar_acordes (bool): Indica si se usan acordes en la melodía.
            proporcion_acordes (float): Proporción de notas que serán acordes (0 a 1).
            rango_octavas (int): Rango de octavas en el que se moverá la melodía.
        """
        self.tonalidad_value = tonalidad_value
        self.tempo = tempo
        self.tonalidad_mayor = tonalidad_value >= 0.5
        self.velocity_media = velocidad_media
        self.densidad_media = densidad_media
        self.caracter_melodico = caracter_melodico
        self.usar_acordes = usar_acordes
        self.proporcion_acordes = proporcion_acordes
        self.rango_octavas = rango_octavas # Nuevo atributo
        

        # Calcula la nota base y la escala de acuerdo con la tonalidad.
        self.nota_base = self._calcular_nota_base()
        self.escala = self._obtener_escala()
        self.sostenidos = self._obtener_sostenidos()
        self.acordes = self._obtener_acordes() # Precalcula los acordes

        # Definir las posibles duraciones de las notas (en términos de tiempos)
        # Cada duración se ajusta a la duración real en segundos, basado en el tempo
        self.duraciones_posibles = [1, 0.5, 0.25, 2, 0.125, 3]  # Negra, corchea, semicorchea, blanca, fusa, redonda
        self.duraciones_en_segundos = {
            1: 60 / self.tempo,  # Negra
            0.5: 60 / (2 * self.tempo),  # Corchea
            0.25: 60 / (4 * self.tempo),  # Semicorchea
            2: 60 / (self.tempo / 2),  # Blanca
            0.125: 60 / (8* self.tempo), # Fusa
            3: 60 / (self.tempo / 4)
        }

        # Para elegir la duración de notas predominante
        self.duracion_media = duracion_media
        self.sigma = sigma
        self.pesos_duracion = self._calcular_pesos(duracion_media)
        
        # Atributos para el manejo de silencios
        self.silencio_maximo = 2  # Por ejemplo, máximo silencio de 2 segundos
        self.probabilidad_silencio = 0.1  # Probabilidad de insertar un silencio

    def _ajustar_espaciado(self, tiempo_inicio, duracion_nota, silencio=False):
        """
        Ajusta el espaciado entre notas basándose en la densidad media.
        Añade la duración de la nota para evitar que se solapen.

        Args:
            tiempo_inicio: Tiempo de inicio de la nota.
            duracion_nota: Duración de la nota.
            silencio (bool): Indica si se inserta un silencio.
        """
        intervalo_base = 60 / self.tempo
        if silencio:
            # El silencio puede ser hasta el doble de la duración de la nota, por ejemplo
            intervalo_silencio = random.uniform(0, self.silencio_maximo)
            return tiempo_inicio + intervalo_silencio
        else:
            intervalo_ajustado = intervalo_base * (1 - self.densidad_media)
            return tiempo_inicio + duracion_nota + intervalo_ajustado

    def _generar_velocity(self):
        # Aseguramos que la velocidad esté en un rango adecuado (0-127 para MIDI)
        # Usa una curva cuadrática para dar más énfasis a los valores cercanos a la media
        velocity = int(np.clip(127 * (self.velocity_media ** 2), 0, 127))
        return velocity
    
    def setVelocityMedia(self, velocity_media):
        self.velocity_media = velocity_media

    def setSigma(self, sigma):
        self.sigma = sigma

    def _calcular_pesos(self, media):
        """Calcula pesos para distribución de duraciones."""
        return {
            d: np.exp(-((d - media) ** 2) / (2 * self.sigma ** 2))
            for d in self.duraciones_posibles
        }

    def _elegir_duracion(self):
        """Elige duración de nota con pesos."""
        return random.choices(
            self.duraciones_posibles,
            weights=[self.pesos_duracion[d] for d in self.duraciones_posibles],
            k=1
        )[0]

    def set_tempo(self, bpm):
        """Establece el tempo."""
        self.tempo = bpm

    def _calcular_nota_base(self):
        """Calcula la nota base."""
        notas = [57, 59, 61, 62, 64, 66, 68,  # a, b, c, d, e, f, g
                60, 62, 64, 65, 67, 69, 71]  # C, D, E, F, G, A, B
        index = int(self.tonalidad_value * 14) - 1
        index = max(0, index)
        return notas[index]

    def _obtener_escala(self):
        """Obtiene la escala."""
        return [0, 2, 4, 5, 7, 9, 11, 12] if self.tonalidad_mayor else [0, 2, 3, 5, 7, 8, 10, 12]

    def _elegir_direccion(self):
        """Elige la dirección de la melodía."""
        if self.caracter_melodico == 0.5:
            return random.choice([1, -1, 0])  # Aleatorio, puede subir, bajar o mantenerse
        elif self.caracter_melodico > 0.5:
            direccion_preferida = 1
        else:
            direccion_preferida = -1
        if random.random() < abs(self.caracter_melodico - 0.5):
            return direccion_preferida
        else:
            return random.choice([1, -1, 0]) # Añadido 0

    def _obtener_sostenidos(self):
        """Devuelve los índices de las notas que deben ser sostenidas."""
        cantidad_sostenidos = int(self.tonalidad_value * 11)
        sostenidos_por_tonalidad = {
            0: [],
            1: [4],
            2: [4, 1],
            3: [4, 1, 5],
            4: [4, 1, 5, 0],
            5: [4, 1, 5, 0, 9],
            6: [4, 1, 5, 0, 9, 2],
            7: [4, 1, 5, 0, 9, 2, 6],
            8: [4, 1, 5, 0, 9, 2, 6, 3],
            9: [4, 1, 5, 0, 9, 2, 6, 3, 7],
            10: [4, 1, 5, 0, 9, 2, 6, 3, 7, 10],
            11: [4, 1, 5, 0, 9, 2, 6, 3, 7, 10, 8],
        }
        return sostenidos_por_tonalidad.get(cantidad_sostenidos, [])
    
    def _obtener_acordes(self):
        """
        Obtiene los acordes para la tonalidad actual.  Simplificado para tríadas mayores y menores.
        """
        if self.tonalidad_mayor:
            return [
                [0, 4, 7],  # Mayor
                [2, 5, 9],  # menor
                [4, 7, 11], # menor
                [5, 9, 12], # Mayor
                [7, 11, 14], # Mayor
                [9, 12, 16], # menor
                [11, 14, 17] # Disminuido
            ]
        else:
            return [
                [0, 3, 7],  # menor
                [2, 5, 8],  # disminuido
                [3, 7, 10], # Mayor
                [5, 8, 12], # menor
                [7, 10, 14], # menor
                [8, 12, 15], # Mayor
                [10, 14, 17] # Mayor
            ]

    def generar_nota(self, nota_relativa, es_acorde=False):
        """
        Genera una nota o acorde, ajustado a la dirección melódica y tonalidad.

        Args:
            nota_relativa (int): El índice de la nota dentro de la escala (0-7).
            es_acorde (bool): Indica si se debe generar un acorde en lugar de una sola nota.

        Returns:
            int o list: El número de nota MIDI o una lista de números de nota MIDI (para acordes).
        """
        # Si no es acorde, genera la nota como antes
        if not es_acorde:
            nota = self.nota_base + self.escala[nota_relativa]
             # Aplica el rango de octavas
            nota += random.randint(0, self.rango_octavas) * 12
            
            if self.tonalidad_mayor:
                if nota_relativa in self.sostenidos:
                    nota += 1
            else:
                indices_validos = [i for i in [0, 3, 5, 7, 10] if i < len(self.escala)]
                notas_a_subir = [self.escala[i] for i in indices_validos[:min(len(self.sostenidos), len(indices_validos))]]
                if self.escala[nota_relativa] in notas_a_subir:
                    nota += 1
            return nota
        
        else: # Si es un acorde
            
            acorde_elegido = self.acordes[nota_relativa % len(self.acordes)] # Usa el acorde correspondiente
            
            # Aplica el rango de octavas a todo el acorde
            acorde_con_octava = [nota + self.nota_base + random.randint(0, self.rango_octavas) * 12 for nota in acorde_elegido]
            
            # Transpone el acorde según la nota relativa de la escala.
            acorde_transpuesto = [nota + self.escala[nota_relativa] for nota in acorde_con_octava]
            
            # Aplica sostenidos si es necesario
            if self.tonalidad_mayor:
                acorde_transpuesto = [nota + 1 if (nota_relativa + i) % 12 in self.sostenidos else nota for i, nota in enumerate(acorde_transpuesto)]
            else:
                indices_validos = [i for i in [0, 3, 5, 7, 10] if i < len(self.escala)]
                notas_a_subir = [self.escala[i] for i in indices_validos[:min(len(self.sostenidos), len(indices_validos))]]
                acorde_transpuesto = [nota + 1 if self.escala[nota_relativa] in notas_a_subir else nota for nota in acorde_transpuesto]
            
            return acorde_transpuesto

    def generar_midi(self, nombre_archivo="tonalidad.mid", instrumento=0, duracion_total=20):
        """
        Genera un archivo MIDI con las notas y duración especificadas.

        Args:
            nombre_archivo (str): Nombre del archivo MIDI.
            instrumento (int): Instrumento MIDI a usar.
            duracion_total (int): Duración total en segundos del archivo midi
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)
        instrumento_midi = pretty_midi.Instrument(program=instrumento, name="Tonalidad")
        tiempo_inicio = 0
        nota_anterior = None  # Para evitar repeticiones consecutivas
        
        while tiempo_inicio < duracion_total:
            # Decide si genera un acorde o una nota simple
            es_acorde = self.usar_acordes and random.random() < self.proporcion_acordes
            
            # Selecciona una nota aleatoria dentro de la escala
            nota_relativa = np.random.choice(len(self.escala))
            
            # Genera la nota o acorde
            nota_actual = self.generar_nota(nota_relativa, es_acorde)
            
             # Evitar repeticiones consecutivas (opcional)
            if nota_anterior is not None and nota_actual == nota_anterior:
                continue
            
            nota_anterior = nota_actual # Guarda la nota actual para la siguiente iteración
            
            # Asignar una duración aleatoria para la nota en términos de segundos
            duracion_aleatoria = self._elegir_duracion()
            duracion_segundos = self.duraciones_en_segundos[duracion_aleatoria]
            
            # Decide si hay un silencio antes de la siguiente nota
            insertar_silencio = random.random() < self.probabilidad_silencio
            
            if es_acorde:
                for nota in nota_actual:
                    nota_midi = pretty_midi.Note(
                        velocity=self._generar_velocity(),
                        pitch=nota,
                        start=tiempo_inicio,
                        end=tiempo_inicio + duracion_segundos
                    )
                    instrumento_midi.notes.append(nota_midi)
            else:
                nota_midi = pretty_midi.Note(
                    velocity=self._generar_velocity(),
                    pitch=nota_actual,
                    start=tiempo_inicio,
                    end=tiempo_inicio + duracion_segundos
                )
                instrumento_midi.notes.append(nota_midi)
                
            # Ajusta el espaciado, incluyendo silencios
            tiempo_inicio = self._ajustar_espaciado(tiempo_inicio, duracion_segundos, silencio=insertar_silencio)

        midi.instruments.append(instrumento_midi)
        midi.write(nombre_archivo)
        print(f"Archivo MIDI generado: {nombre_archivo}")

def generar_parametros_cielo(filename):
    """
    Genera diccionarios de parámetros MIDI basados en el tipo de cielo.

    Args:
        tipo_cielo (str): El tipo de cielo a representar.

    Returns:
        dict: Un diccionario con los parámetros MIDI.
    """
    resultado = {}
    if filename == "noche.jpeg":
        resultado = {
            'tonalidad_value': 0.173401857647427,
            'tempo': 60.0,
            'duracion_media': 2.332972439236705,
            'sigma': 0.794629937748768,
            'velocidad_media': 0.7058401363070101,
            'densidad_media': 0.49661800355549396,
            'caracter_melodico': 0.5853888627107341,
            'usar_acordes': 1.0,
            'proporcion_acordes': 0.5210013624425488,
            'rango_octavas': 1.0
        }
    elif filename == "lluvia.jpeg":
        resultado = {
            'tonalidad_value': 0.18511513287057274,
            'tempo': 104.0,
            'duracion_media': 1.6200326409067443,
            'sigma': 0.5971889966624578,
            'velocidad_media': 0.7338613530265905,
            'densidad_media': 0.6304160343296756,
            'caracter_melodico': 0.631621221984135,
            'usar_acordes': 0.0,
            'proporcion_acordes': 0.20133991297536086,
            'rango_octavas': 1.0
        }
    elif filename == "soleado.jpeg":
        resultado = {
            'tonalidad_value': 0.6038577728765497,
            'tempo': 103.0,
            'duracion_media': 1.0922498772548028,
            'sigma': 0.328964091631769,
            'velocidad_media': 0.8672986591368046,
            'densidad_media': 0.6260613316977481,
            'caracter_melodico': 0.7436377055721527,
            'usar_acordes': 1.0,
            'proporcion_acordes': 0.6136732727495986,
            'rango_octavas': 2.0
        }
    elif filename == "tormenta.jpeg":
        resultado = {
            'tonalidad_value': 0.1512525447110794,
            'tempo': 142.0,
            'duracion_media': 2.000874726495423,
            'sigma': 0.8016621408469216,
            'velocidad_media': 0.7626121104479034,
            'densidad_media': 0.8491398506454925,
            'caracter_melodico': 0.6451232638991419,
            'usar_acordes': 0.0,
            'proporcion_acordes': 0.14357213470301866,
            'rango_octavas': 1.0
        }
    else:
        resultado = None


    return resultado

if __name__ == '__main__':
    tipos_de_cielo = ["noche.jpeg", "lluvia.jpeg", "tormenta.jpeg", "soleado.jpeg"
    
]
    for tipo in tipos_de_cielo:
        parametros = generar_parametros_cielo(tipo)
        print(f"Parámetros para cielo '{tipo}': {parametros}")
        params_amidi = ParamsAMIDI(**parametros)
        params_amidi.generar_midi(nombre_archivo=f"{tipo}.mid")