-- GUARDA EL ITEM MIDI SELECIONADO --

-- El editor MIDI del item debe estar abierto
local editor = reaper.MIDIEditor_GetActive()
if not editor then
  reaper.ShowMessageBox("Abre primero el ítem en el editor MIDI.", "Editor MIDI no activo", 0)
  return
end

-- Ejecuta la acción "File: Export contents as .MID" (solo disponible en el editor MIDI)
-- Se abrirá una ventana para guardar el archivo
reaper.MIDIEditor_OnCommand(editor, 40038)
