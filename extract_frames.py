import cv2
import os

# Carpeta de videos (ajusta según el tipo: originales o manipulados)
input_folder = ''  # Ingresar las rutas del dataset
output_folder = ''

os.makedirs(output_folder, exist_ok=True)

for video_name in os.listdir(input_folder):
    if not video_name.endswith('.mp4'):
        continue
    video_path = os.path.join(input_folder, video_name)
    video_id = os.path.splitext(video_name)[0]
    video_output_dir = os.path.join(output_folder, video_id)
    os.makedirs(video_output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Guarda cada frame como imagen
        frame_filename = os.path.join(video_output_dir, f"{video_id}_frame{frame_idx:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1
    cap.release()
    print(f"Procesado: {video_name} -> {frame_idx} frames extraídos")
print("¡Listo!")
