import cv2
import requests

ROBOFLOW_API_KEY = ""  # ¡Reemplaza con tu clave real!
ROBOFLOW_MODEL_ID = "coco-person"  # Modelo pre-entrenado para detección de personas
ROBOFLOW_API_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"
CONFIDENCE_THRESHOLD = 0.5  # Umbral de confianza para considerar una detección válida

def capturar_imagen():
    """Captura una imagen desde la cámara web."""
    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise IOError("No se puede acceder a la cámara web")

        ret, frame = cam.read()
        if not ret:
            raise IOError("No se pudo capturar la imagen")

        cv2.imwrite("imagen_capturada.jpg", frame)
        print("Imagen capturada como 'imagen_capturada.jpg'")
        cam.release()
        return "imagen_capturada.jpg"
    except IOError as e:
        print(f"Error al capturar la imagen: {e}")
        return None

def detectar_persona_con_api(ruta_imagen):
    """Detecta si hay una persona en la imagen utilizando la API de Roboflow."""
    if ruta_imagen is None:
        return "No se proporcionó una imagen."

    try:
        with open(ruta_imagen, 'rb') as image_file:
            files = {'file': ('imagen.jpg', image_file, 'image/jpeg')}
            params = {'api_key': ROBOFLOW_API_KEY}  # La clave de API va en los parámetros

            response = requests.post(
                f"{ROBOFLOW_API_URL}",  # URL base del modelo
                files=files,
                params=params  # Enviamos la clave de API como parámetro
            )
            response.raise_for_status()  # Lanza una excepción para códigos de error HTTP
            results = response.json()

            if results and results.get('predictions'):
                personas_detectadas = False
                for prediction in results['predictions']:
                    if prediction.get('class') == 'person' and prediction.get('confidence', 0) > CONFIDENCE_THRESHOLD:
                        personas_detectadas = True
                        break

                if personas_detectadas:
                    return "Se detectó una persona en la imagen."
                else:
                    return "No se detectó una persona con alta confianza en la imagen."
            else:
                return "No se encontraron predicciones en la respuesta de la API."

    except requests.exceptions.RequestException as e:
        return f"Error al contactar la API de Roboflow: {e}"
    except FileNotFoundError:
        return f"No se encontró la imagen en la ruta: {ruta_imagen}"
    except Exception as e:
        return f"Ocurrió un error al procesar la respuesta de la API: {e}"

if __name__ == "__main__":
    print("Iniciando el programa de detección de personas...")
    ruta_imagen = capturar_imagen()

    if ruta_imagen:
        resultado_deteccion = detectar_persona_con_api(ruta_imagen)
        print("Resultado de la detección:", resultado_deteccion)
    else:
        print("No se pudo continuar con la detección debido a un problema con la captura de la imagen.")

print("Programa finalizado.")