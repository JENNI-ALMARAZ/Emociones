# Primero, importa os para configurar el nivel de log
import os
# Establece el nivel de log para suprimir advertencias
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = todas, 1 = advertencias, 2 = errores

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import time
from deepface import DeepFace
import cv2
import mediapipe as mp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

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
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Función para procesar y dibujar puntos faciales específicos con MediaPipe
def draw_landmarks_with_mediapipe(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Inicializar MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Índices refinados para ojos, cejas, boca y nariz
                specific_indices = {
                    "Cejas": [70, 107, 336],
                    "Ojo derecho": [33, 133],
                    "Ojo izquierdo": [263, 362],
                    "Boca": [78, 308, 61, 291],  # Extremos, labio superior e inferior
                    "Nariz": [1]
                }

                # Dibujar puntos y mostrar coordenadas
                for feature, indices in specific_indices.items():
                    for idx in indices:
                        if idx < len(face_landmarks.landmark):
                            x = int(face_landmarks.landmark[idx].x * image.shape[1])
                            y = int(face_landmarks.landmark[idx].y * image.shape[0])

                            # Dibujar el punto
                            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Guardar la imagen procesada
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(processed_path, image)
    return processed_path

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
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Simular un retraso para procesamiento
            time.sleep(2)

            # Procesar la imagen usando DeepFace para obtener la emoción
            result = DeepFace.analyze(
                img_path=filepath,
                actions=["emotion"],
                enforce_detection=False
            )

            # Obtener la emoción dominante y traducirla
            emotion_english = result[0]["dominant_emotion"]
            emotion_spanish = emotion_translation.get(emotion_english, "Emoción desconocida")

            # Procesar puntos faciales con MediaPipe
            processed_image_path = draw_landmarks_with_mediapipe(filepath)

            return render_template(
                'result.html',
                original_image=filepath,
                processed_image=processed_image_path,
                emotion=emotion_spanish
            )
        except Exception as e:
            return f"Error processing the image: {str(e)}"
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Crear carpetas si no existen
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)
