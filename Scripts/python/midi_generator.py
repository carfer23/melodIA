import pretty_midi
import numpy as np
import unittest
import os
import random

class ParamsAMIDI:

    def __init__(self, tonalidad_value, tempo=120):
        """
        Inicializa la clase con los parámetros de tonalidad, tempo, y duración total del MIDI.

        Args:
            tonalidad_value (float): Un valor entre 0 y 1 que representa la tonalidad.
            tempo (int): Tempo en BPM (por defecto 120).
            duracion_total (float): Duración total del MIDI en segundos.
        """
        self.tonalidad_value = tonalidad_value
        self.tempo = tempo
        self.tonalidad_mayor = tonalidad_value > 0.5

        # Calcula la nota base y la escala de acuerdo con la tonalidad.
        self.nota_base = self._calcular_nota_base()
        self.escala = self._obtener_escala()
        self.sostenidos = self._obtener_sostenidos()

        # Definir las posibles duraciones de las notas (en términos de tiempos)
        # Cada duración se ajusta a la duración real en segundos, basado en el tempo
        self.duraciones_posibles = [1, 0.5, 0.25, 2]  # Negra, corchea, semicorchea, blanca
        self.duraciones_en_segundos = {
            1: 60 / self.tempo,  # Negra
            0.5: 60 / (2 * self.tempo),  # Corchea
            0.25: 60 / (4 * self.tempo),  # Semicorchea
            2: 60 / (self.tempo / 2)  # Blanca
        }

    def set_tempo(self, bpm):
        self.tempo = bpm

    def _calcular_nota_base(self):
        notas = [57, 59, 61, 62, 64, 66, 68,  # a, b, c, d, e, f, g
                60, 62, 64, 65, 67, 69, 71]  # C, D, E, F, G, A, B

        index = int(self.tonalidad_value * 14) - 1
        index = max(0, index)

        return notas[index]

    def _obtener_escala(self):
        return [0, 2, 4, 5, 7, 9, 11, 12] if self.tonalidad_mayor else [0, 2, 3, 5, 7, 8, 10, 12]

    def _obtener_sostenidos(self):
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

    def generar_nota(self, nota_relativa):
        nota = self.nota_base + self.escala[nota_relativa]
        if self.tonalidad_mayor:
            if nota_relativa in self.sostenidos:
                nota += 1
        else:
            indices_validos = [i for i in [0, 3, 5, 7, 10] if i < len(self.escala)]
            notas_a_subir = [self.escala[i] for i in indices_validos[:min(len(self.sostenidos), len(indices_validos))]]
            if self.escala[nota_relativa] in notas_a_subir:
                nota += 1
        return nota

    def generar_midi(self, nombre_archivo="tonalidad.mid", instrumento=0, duracion_total=10):
        """
        Genera un archivo MIDI con las notas y duración especificadas.

        Args:
            nombre_archivo (str): Nombre del archivo MIDI.
            instrumento (int): Instrumento MIDI a usar.
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)
        instrumento_midi = pretty_midi.Instrument(program=instrumento, name="Tonalidad")
        tiempo_inicio = 0

        while tiempo_inicio < duracion_total:
            # Seleciona una nota aleatoria dentro de la escala
            nota_relativa = np.random.choice(len(self.escala))
            nota_midi = pretty_midi.Note(
                velocity=64,
                pitch=self.generar_nota(nota_relativa),
                start=tiempo_inicio,
                end=tiempo_inicio
            )
            # Asignar una duración aleatoria para la nota en términos de segundos
            duracion_aleatoria = random.choice(self.duraciones_posibles)
            duracion_segundos = self.duraciones_en_segundos[duracion_aleatoria]
            nota_midi.end = tiempo_inicio + duracion_segundos

            instrumento_midi.notes.append(nota_midi)

            tiempo_inicio += duracion_segundos

        midi.instruments.append(instrumento_midi)
        midi.write(nombre_archivo)
        print(f"Archivo MIDI generado: {nombre_archivo}")

class TestParamsAMIDI(unittest.TestCase):
    def test_calcular_nota_base(self):
        self.assertEqual(ParamsAMIDI(0.07142857142857142)._calcular_nota_base(), 57)  # a 1/14 etc
        self.assertEqual(ParamsAMIDI(0.14285714285714285)._calcular_nota_base(), 59)  # b
        self.assertEqual(ParamsAMIDI(0.21428571428571427)._calcular_nota_base(), 61)  # c
        self.assertEqual(ParamsAMIDI(0.2857142857142857)._calcular_nota_base(), 62)   # d
        self.assertEqual(ParamsAMIDI(0.35714285714285715)._calcular_nota_base(), 64)  # e
        self.assertEqual(ParamsAMIDI(0.42857142857142855)._calcular_nota_base(), 66)  # f
        self.assertEqual(ParamsAMIDI(0.5)._calcular_nota_base(), 68)                  # g
        self.assertEqual(ParamsAMIDI(0.5714285714285714)._calcular_nota_base(), 60)   # C
        self.assertEqual(ParamsAMIDI(0.6428571428571429)._calcular_nota_base(), 62)   # D
        self.assertEqual(ParamsAMIDI(0.7142857142857143)._calcular_nota_base(), 64)   # E
        self.assertEqual(ParamsAMIDI(0.7857142857142857)._calcular_nota_base(), 65)   # F
        self.assertEqual(ParamsAMIDI(0.8571428571428571)._calcular_nota_base(), 67)   # G
        self.assertEqual(ParamsAMIDI(0.9285714285714286)._calcular_nota_base(), 69)   # A
        self.assertEqual(ParamsAMIDI(1.0)._calcular_nota_base(), 71)                  # B

    def test_obtener_escala(self):
        self.assertEqual(ParamsAMIDI(0.7).escala, [0, 2, 4, 5, 7, 9, 11, 12])
        self.assertEqual(ParamsAMIDI(0.2).escala, [0, 2, 3, 5, 7, 8, 10, 12])

    def test_obtener_sostenidos(self):
        self.assertEqual(ParamsAMIDI(0.0).sostenidos, [])
        self.assertEqual(ParamsAMIDI(0.5).sostenidos, [4, 1, 5, 0, 9])
        self.assertEqual(ParamsAMIDI(0.2).sostenidos, [4, 1])

    def test_generar_nota(self):
        self.assertEqual(ParamsAMIDI(0.0).generar_nota(0), 57)
        self.assertEqual(ParamsAMIDI(0.0).generar_nota(4), 64)

        self.assertEqual(ParamsAMIDI(0.714285).generar_nota(4), 70)
        self.assertEqual(ParamsAMIDI(0.714285).generar_nota(6), 74)

        self.assertEqual(ParamsAMIDI(0.428571).generar_nota(0), 65)
        self.assertEqual(ParamsAMIDI(0.428571).generar_nota(4), 71)

    def test_generar_midi(self):
        tonalidad_a_midi = ParamsAMIDI(0.7)
        nombre_archivo = "test_generar_midi.mid"
        tonalidad_a_midi.generar_midi(nombre_archivo=nombre_archivo)
        self.assertTrue(os.path.exists(nombre_archivo))
        os.remove(nombre_archivo)

    # TESTS DURACION NOTAS
    def setUp(self):
        # Configuración básica para los tests
        self.params = ParamsAMIDI(tonalidad_value=0.5, tempo=120, duracion_total=10)

    def test_generar_duracion_aleatoria(self):
        """
        Testea si las duraciones de las notas son aleatorias y están ajustadas al tempo.
        """
        duraciones = []

        # Generamos 100 notas para verificar las duraciones
        for _ in range(100):
            tiempo_inicio = 0
            while tiempo_inicio < self.params.duracion_total:
                # Generar una duración aleatoria y agregarla al tiempo de inicio
                duracion_aleatoria = random.choice(self.params.duraciones_posibles)
                duracion_segundos = self.params.duraciones_en_segundos[duracion_aleatoria]
                duraciones.append(duracion_segundos)
                tiempo_inicio += duracion_segundos

        # Verificamos si las duraciones son las esperadas
        for duracion in duraciones:
            self.assertIn(duracion, self.params.duraciones_en_segundos.values())

    def test_generar_midi_duracion_total(self):
        """
        Testea si la duración total del archivo MIDI es cercana a la duración total especificada.
        """
        nombre_archivo = "test_duracion_total.mid"
        self.params.generar_midi(nombre_archivo)

        # Cargar el archivo MIDI generado
        midi = pretty_midi.PrettyMIDI(nombre_archivo)

        # Calcular la duración total del archivo MIDI
        duracion_total_generada = midi.get_end_time()

        # Comprobar que la duración generada está dentro de un rango aceptable de ±0.5 segundos
        self.assertAlmostEqual(duracion_total_generada, self.params.duracion_total, delta=0.5)

    def test_temperatura_tiempos_validos(self):
        """
        Testea si los tiempos de inicio y final de las notas están dentro de los límites de la duración total.
        """
        nombre_archivo = "test_temperatura_tiempos.mid"
        self.params.generar_midi(nombre_archivo)

        # Cargar el archivo MIDI generado
        midi = pretty_midi.PrettyMIDI(nombre_archivo)
        
        # Verificar que todas las notas tienen tiempos válidos
        for nota in midi.instruments[0].notes:
            self.assertGreaterEqual(nota.start, 0)  # La nota debe empezar en o después del tiempo 0
            self.assertLessEqual(nota.end, self.params.duracion_total)  # La nota debe terminar dentro de la duración total

    def test_duraciones_correctas_segundos(self):
        """
        Testea si las duraciones de las notas en segundos corresponden con los valores correctos ajustados al tempo.
        """
        duraciones_validas = {
            1: 60 / self.params.tempo,  # Negra
            0.5: 60 / (2 * self.params.tempo),  # Corchea
            0.25: 60 / (4 * self.params.tempo),  # Semicorchea
            2: 60 / (self.params.tempo / 2)  # Blanca
        }

        for duracion in self.params.duraciones_posibles:
            duracion_segundos = self.params.duraciones_en_segundos[duracion]
            self.assertEqual(duracion_segundos, duraciones_validas[duracion])


if __name__ == '__main__':
    #unittest.main()

    # Ejemplo 1: C mayor, tempo lento
    params_c_mayor = ParamsAMIDI(0.0, tempo=60)
    params_c_mayor.generar_midi(nombre_archivo="c_mayor_lento.mid")

    # Ejemplo 2: G mayor, tempo rápido
    params_g_mayor = ParamsAMIDI(0.5, tempo=180)
    params_g_mayor.generar_midi(nombre_archivo="g_mayor_rapido.mid")

    # Ejemplo 3: A menor, tempo moderado
    params_a_menor = ParamsAMIDI(0.142857, tempo=120)
    params_a_menor.generar_midi(nombre_archivo="a_menor.mid")

    # Ejemplo 4: D mayor, tempo lento
    params_d_mayor = ParamsAMIDI(0.285714, tempo=80)
    params_d_mayor.generar_midi(nombre_archivo="d_mayor_lento.mid")

    # Ejemplo 5: E menor, tempo rápido
    params_e_menor = ParamsAMIDI(0.428571, tempo=160)
    params_e_menor.generar_midi(nombre_archivo="e_menor_rapido.mid")

    # Ejemplo 6: B menor, tempo muy lento
    params_b_menor = ParamsAMIDI(0.928571, tempo=40)
    params_b_menor.generar_midi(nombre_archivo="b_menor_muy_lento.mid")

    # Ejemplo 7: F mayor, tempo variado
    params_f_mayor = ParamsAMIDI(0.285714, tempo=110)
    params_f_mayor.generar_midi(nombre_archivo="f_mayor.mid")

    # Ejemplo 8: C menor, tempo muy rápido
    params_c_menor = ParamsAMIDI(0.857143, tempo=200)
    params_c_menor.generar_midi(nombre_archivo="c_menor_muy_rapido.mid")

    # Ejemplo 9: C mayor, cambiar tempo dinámicamente
    params_c_mayor = ParamsAMIDI(0.0, tempo=60)
    params_c_mayor.generar_midi(nombre_archivo="c_mayor_lento.mid")

    params_c_mayor.set_tempo(120)
    params_c_mayor.generar_midi(nombre_archivo="c_mayor_moderado.mid")

    params_c_mayor.set_tempo(180)
    params_c_mayor.generar_midi(nombre_archivo="c_mayor_rapido.mid")


