"""Script de prueba para extender un archivo MIDI con el modelo preentrenado de Magenta MelodyRNN.
Usamos este script para probar a llamarlo desde un script LUA de ReaScipt pero no conseguimos que funcionara.
El script LUA de prueba se encuentra en proyecto_reaper/Scripts/prueba-nofunciona.lua"""

from note_seq import midi_file_to_note_sequence
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2
import note_seq

# Ruta al archivo MIDI exportado desde Reaper
input_midi_path = "Media/hola.mid"

# Convertimos el archivo MIDI a NoteSequence (formato usado por Magenta)
try:
    input_sequence = midi_file_to_note_sequence(input_midi_path)
    print("MIDI cargado correctamente. Número de notas:", len(input_sequence.notes))
except Exception as e:
    print("Error al cargar el archivo MIDI:", e)

# Rutas
bundle_path = "../../models/basic_rnn.mag"
output_path = "../../Media/hola_generated.mid"

# Cargar modelo
bundle = sequence_generator_bundle.read_bundle_file(bundle_path)
generator_map = melody_rnn_sequence_generator.get_generator_map()
melody_rnn = generator_map['basic_rnn'](checkpoint=None, bundle=bundle)
melody_rnn.initialize()

num_steps = 128 # change this for shorter or longer sequences
temperature = 1.0 # the higher the temperature the more random the sequence.

# Set the start time to begin on the next step after the last note ends.
last_end_time = (max(n.end_time for n in input_sequence.notes)
                  if input_sequence.notes else 0)
qpm = input_sequence.tempos[0].qpm 
seconds_per_step = 60.0 / qpm / melody_rnn.steps_per_quarter
total_seconds = num_steps * seconds_per_step

generator_options = generator_pb2.GeneratorOptions()
generator_options.args['temperature'].float_value = temperature
generate_section = generator_options.generate_sections.add(
  start_time=last_end_time + seconds_per_step,
  end_time=total_seconds)

generated_sequence = melody_rnn.generate(input_sequence, generator_options)

note_seq.sequence_proto_to_midi_file(generated_sequence, output_path)
