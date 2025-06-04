import os
from PIL import Image
from mtcnn import MTCNN
import numpy as np

input_folder = '' #poner ruta de los frames extraido
output_folder = ''

os.makedirs(output_folder, exist_ok=True)

detector = MTCNN()

# Loop sobre los videos (carpetas de frames)
for video_id in os.listdir(input_folder):
    video_frame_dir = os.path.join(input_folder, video_id)
    if not os.path.isdir(video_frame_dir):
        continue

    # Carpeta de salida para las caras de este video
    video_output_dir = os.path.join(output_folder, video_id)
    os.makedirs(video_output_dir, exist_ok=True)

    existing_faces = [f for f in os.listdir(video_output_dir) if f.endswith('.jpg')]
    if len(existing_faces) > 0:
        print(f"Saltando video {video_id}: ya procesado ({len(existing_faces)} caras).")
        continue

    # Procesa todos los frames
    for frame_name in os.listdir(video_frame_dir):
        frame_path = os.path.join(video_frame_dir, frame_name)
        image = Image.open(frame_path).convert('RGB')
        image_array = np.asarray(image)

        faces = detector.detect_faces(image_array)
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            cropped_face = image.crop((x, y, x + w, y + h))
            face_filename = os.path.join(video_output_dir, f"{os.path.splitext(frame_name)[0]}_face{i}.jpg")
            cropped_face.save(face_filename)

    print(f"Caras extraídas del video: {video_id}")

print("¡Extracción de caras completada!")
