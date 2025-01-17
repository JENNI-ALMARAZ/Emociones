from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import time
from deepface import DeepFace
import cv2
import base64
import numpy as np
import mediapipe as mp
from PIL import Image
import io

app = Flask(__name__)

# Diccionario para traducir emociones
emotion_translation = {
    "angry": "Enojado",
    "disgust": "Disgusto",
    "fear": "Miedo",
    "happy": "Feliz",
    "sad": "Triste",
    "surprise": "Sorpresa",
    "neutral": "Neutral"
}

# Verifica si el archivo es permitido
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Función para redimensionar imágenes grandes
def resize_image(image, max_width=800, max_height=800):
    """
    Redimensiona una imagen para que no exceda el tamaño máximo permitido.
    """
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

# Función para procesar y dibujar puntos faciales específicos con MediaPipe
def draw_landmarks_with_mediapipe(image):
    mp_face_mesh = mp.solutions.face_mesh
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Inicializar MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Índices refinados para ojos, cejas, boca y nariz
                specific_indices = {
                    "Cejas izquierda": [65, 55, 52, 46],
                    "Cejas derecha": [295, 285, 282, 276],
                    "Ojo derecho": [33, 133, 159, 145],
                    "Ojo izquierdo": [263, 362, 386, 374],
                    "Boca": [61, 291, 13, 14],
                    "Nariz": [1]
                }

                for indices in specific_indices.values():
                    for idx in indices:
                        if idx < len(face_landmarks.landmark):
                            x = int(face_landmarks.landmark[idx].x * image.shape[1])
                            y = int(face_landmarks.landmark[idx].y * image.shape[0])
                            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        # Leer la imagen directamente desde el archivo cargado
        file_bytes = file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Redimensionar la imagen si es necesario
        img = resize_image(img)

        try:
            time.sleep(2)

            # Procesar la imagen usando DeepFace para obtener la emoción
            result = DeepFace.analyze(
                img_path=img,
                actions=["emotion"],
                enforce_detection=False
            )

            # Obtener la emoción dominante y traducirla
            emotion_english = result[0]["dominant_emotion"]
            emotion_spanish = emotion_translation.get(emotion_english, "Emoción desconocida")

            # Convertir la imagen original a base64
            _, img_encoded_original = cv2.imencode('.jpg', img)
            original_image_base64 = base64.b64encode(img_encoded_original).decode('utf-8')

            # Procesar puntos faciales con MediaPipe
            img_with_landmarks = draw_landmarks_with_mediapipe(img.copy())

            # Convertir la imagen procesada (con puntos faciales) a base64
            _, img_encoded_processed = cv2.imencode('.jpg', img_with_landmarks)
            processed_image_base64 = base64.b64encode(img_encoded_processed).decode('utf-8')

            return render_template(
                'result.html',
                original_image=original_image_base64,
                processed_image=processed_image_base64,
                emotion=emotion_spanish
            )
        except Exception as e:
            return f"Error processing the image: {str(e)}"
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
