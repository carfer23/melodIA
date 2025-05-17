from PIL import Image

def cargar_imagen(ruta):
    return Image.open(ruta).convert("RGB")
