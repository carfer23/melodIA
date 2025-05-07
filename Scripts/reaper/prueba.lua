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


-- LLAMA AL SCRIPT PYTHON --

local project_path = reaper.GetProjectPath("")  -- ruta del proyecto REAPER
local script_path = project_path .. "/Scripts/python/melodyrnn.py"

os.execute("python \"" .. script_path .. "\"")

-- Ruta absoluta del archivo MIDI generado
local file_path = project_path .. "/Media/hola_generated.mid"

-- Espera a que el nuevo archivo se genere
local function waitForFile(file_path, timeout_sec)
  local start_time = reaper.time_precise()
  while not reaper.file_exists(file_path) do
    reaper.defer(function() end)  -- deja respirar a REAPER
    if reaper.time_precise() - start_time > timeout_sec then
      return false  -- timeout
    end
  end
  return true
end

if not waitForFile(file_path, 10) then
  reaper.ShowMessageBox("Archivo MIDI no disponible tras esperar.", "Error", 0)
  return
end

-- INSERTA EL MIDI EXTENDIDO EN REAPER --

-- Selecciona el primer track (puedes modificar esto según necesites)
local track = reaper.GetTrack(0, 0) -- primer track

if track then
  -- Crea un nuevo source desde el archivo MIDI
  local source = reaper.PCM_Source_CreateFromFile(file_path)

  if source then
    -- Obtiene la duración del archivo MIDI
    local source_len = reaper.GetMediaSourceLength(source)

    -- Crea nuevo media item en el track
    local new_item = reaper.AddMediaItemToTrack(track)

    -- Establece la posición (en el cursor) y la duración del item
    local cursor_pos = reaper.GetCursorPosition()
    reaper.SetMediaItemPosition(new_item, cursor_pos, false)
    reaper.SetMediaItemInfo_Value(new_item, "D_LENGTH", source_len)

    -- Crea un nuevo take que use el source MIDI
    local take = reaper.AddTakeToMediaItem(new_item)
    reaper.SetMediaItemTake_Source(take, source)

    -- Actualiza la vista
    reaper.UpdateArrange()
  else
    reaper.ShowMessageBox("No se pudo cargar el archivo MIDI:\n" .. file_path, "Error", 0)
  end
else
  reaper.ShowMessageBox("No hay ninguna pista disponible.", "Error", 0)
end
