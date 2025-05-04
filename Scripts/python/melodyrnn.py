from note_seq import midi_file_to_note_sequence

# Ruta al archivo MIDI exportado desde Reaper
input_midi_path = "Media/hola.MID"

# Convertimos el archivo MIDI a NoteSequence (formato usado por Magenta)
try:
    input_sequence = midi_file_to_note_sequence(input_midi_path)
    print("MIDI cargado correctamente. Número de notas:", len(input_sequence.notes))
except Exception as e:
    print("Error al cargar el archivo MIDI:", e)
