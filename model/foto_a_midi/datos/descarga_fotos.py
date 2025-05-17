import os
import requests

API_KEY = '9KqFI5k0CtGFM1BssZ94KYpKovCBmzSuzCMspbL6TcPx2PflMM1jEmJm'
HEADERS = {'Authorization': API_KEY}
SEARCH_TERMS = [
    "thunderstorm", "rainy sky", "bright sunny sky",
    "rainbow", "starry night", "clouds",
    "sunset", "snow"
]
PER_PAGE = 300
DOWNLOAD_DIR = "pexels_cielos_varios"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def descargar_imagen(url, path):
    r = requests.get(url)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            f.write(r.content)

def descargar_imagenes():
    imagenes_descargadas = 0
    for term in SEARCH_TERMS:
        print(f"Buscando imágenes para: {term}")
        url = f"https://api.pexels.com/v1/search?query={term.replace(' ', '+')}&per_page={PER_PAGE}&page=1"
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Error en la búsqueda para {term}: {response.status_code}")
            continue
        data = response.json()
        fotos = data.get('photos', [])
        for i, foto in enumerate(fotos):
            img_url = foto['src']['large2x']
            nombre_archivo = f"{term.replace(' ', '_')}_{i}.jpg"
            ruta = os.path.join(DOWNLOAD_DIR, nombre_archivo)
            descargar_imagen(img_url, ruta)
            imagenes_descargadas += 1
            print(f"Descargada: {nombre_archivo}")
    print(f"Total imágenes descargadas: {imagenes_descargadas}")

if __name__ == '__main__':
    descargar_imagenes()
