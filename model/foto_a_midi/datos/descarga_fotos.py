import os
import requests

class PexelsDownloader:
    """
    Clase para descargar imágenes desde la API de Pexels.

    Proporciona métodos estáticos para buscar y descargar imágenes
    basadas en términos de búsqueda, guardándolas en una carpeta local.
    """
    # API key para autenticación con Pexels
    API_KEY = '9KqFI5k0CtGFM1BssZ94KYpKovCBmzSuzCMspbL6TcPx2PflMM1jEmJm'
    HEADERS = {'Authorization': API_KEY}

    @staticmethod
    def descargar_imagen(url, path):
        """Descarga una imagen desde una URL y la guarda en la ruta especificada."""
        r = requests.get(url)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                f.write(r.content)

    @staticmethod
    def descargar_imagenes(search_terms, per_page=300, download_dir="pexels_cielos_varios"):
        """
        Descarga imágenes desde Pexels para cada término de búsqueda.

        Args:
            search_terms (list): Lista de términos para buscar imágenes.
            per_page (int): Cantidad máxima de imágenes por término.
            download_dir (str): Carpeta donde se guardarán las imágenes.

        Returns:
            int: Total de imágenes descargadas.
        """
        os.makedirs(download_dir, exist_ok=True)
        imagenes_descargadas = 0

        for term in search_terms:
            print(f"Buscando imágenes para: {term}")
            url = f"https://api.pexels.com/v1/search?query={term.replace(' ', '+')}&per_page={per_page}&page=1"
            response = requests.get(url, headers=PexelsDownloader.HEADERS)
            if response.status_code != 200:
                print(f"Error en la búsqueda para {term}: {response.status_code}")
                continue
            data = response.json()
            fotos = data.get('photos', [])
            for i, foto in enumerate(fotos):
                img_url = foto['src']['large2x']
                nombre_archivo = f"{term.replace(' ', '_')}_{i}.jpg"
                ruta = os.path.join(download_dir, nombre_archivo)
                PexelsDownloader.descargar_imagen(img_url, ruta)
                imagenes_descargadas += 1
                print(f"Descargada: {nombre_archivo}")

        print(f"Total imágenes descargadas: {imagenes_descargadas}")
        return imagenes_descargadas

