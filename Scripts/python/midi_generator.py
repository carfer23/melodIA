import pretty_midi
import numpy as np
import unittest
import os
import math

class ParamsAMIDI:
    def __init__(self, tonalidad_value, tempo=120):
        self.tonalidad_value = tonalidad_value
        self.tempo = tempo
        self.tonalidad_mayor = tonalidad_value > 0.5
        self.nota_base = self._calcular_nota_base()
        self.escala = self._obtener_escala()
        self.sostenidos = self._obtener_sostenidos()

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

    def generar_midi(self, nombre_archivo="tonalidad.mid", instrumento=0, duracion_segundos=10):
        duracion_negra = 60 / self.tempo
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)
        instrumento_midi = pretty_midi.Instrument(program=instrumento, name="Tonalidad")
        tiempo_inicio = 0
        while tiempo_inicio < duracion_segundos:
            nota_relativa = np.random.choice(len(self.escala))
            nota_midi = pretty_midi.Note(
                velocity=64,
                pitch=self.generar_nota(nota_relativa),
                start=tiempo_inicio,
                end=tiempo_inicio + duracion_negra
            )
            instrumento_midi.notes.append(nota_midi)
            tiempo_inicio += duracion_negra
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


