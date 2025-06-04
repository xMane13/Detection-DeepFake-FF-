import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from sklearn.metrics import classification_report, confusion_matrix

# Rutas a los modelos .h5
models_paths = [
    '/home/manuel.munoz__yachaytech.edu.ec/xception_ffpp_best.h5',            # FaceSwap (general)
    '/home/manuel.munoz__yachaytech.edu.ec/xception_face2face_best.h5',
    '/home/manuel.munoz__yachaytech.edu.ec/xception_deepfakes_best.h5',
    '/home/manuel.munoz__yachaytech.edu.ec/xception_neuraltextures_best.h5'
]

# Diccionario con los sets de test
test_roots = {
    'FaceSwap': '/home/manuel.munoz__yachaytech.edu.ec/FFPP_data/faces_split/test',
    'Face2Face': '/home/manuel.munoz__yachaytech.edu.ec/FFPP_data/faces_split_face2face/test',
    'Deepfakes': '/home/manuel.munoz__yachaytech.edu.ec/FFPP_data/faces_split_deepfakes/test',
    'NeuralTextures': '/home/manuel.munoz__yachaytech.edu.ec/FFPP_data/faces_split_neuraltextures/test'
}
classes = ['real', 'fake']

models = [load_model(path) for path in models_paths]

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    arr = image.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

for dataset_name, test_root in test_roots.items():
    print(f"\n\n=========== Evaluando Ensemble en set: {dataset_name} ===========")
    img_paths = []
    labels = []
    for idx, cls in enumerate(classes):
        folder = os.path.join(test_root, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith('.jpg'):
                img_paths.append(os.path.join(folder, fname))
                labels.append(idx)  # 0=real, 1=fake

    y_true = np.array(labels)
    y_preds = []
    y_majority = []
    for i, img_path in enumerate(img_paths):
        arr = preprocess_img(img_path)
        preds = [model.predict(arr, verbose=0)[0,0] for model in models]
        mean_pred = np.mean(preds)
        majority_vote = int(np.sum(np.array(preds) > 0.5) >= (len(preds)//2 + 1))
        y_preds.append(mean_pred)
        y_majority.append(majority_vote)
        if (i+1) % 500 == 0 or (i+1) == len(img_paths):
            print(f'Procesadas {i+1}/{len(img_paths)} imágenes')

    y_pred_binary = (np.array(y_preds) > 0.5).astype(int)

    print("\n--- Ensemble: Promedio de probabilidades (umbral 0.5) ---")
    print(classification_report(y_true, y_pred_binary, target_names=classes))
    print("Matriz de confusión:\n", confusion_matrix(y_true, y_pred_binary))

    print("\n--- Ensemble: Votación mayoritaria ---")
    print(classification_report(y_true, y_majority, target_names=classes))
    print("Matriz de confusión:\n", confusion_matrix(y_true, y_majority))

    # Guarda el reporte para cada dataset
    with open(f'ensemble_classification_report_{dataset_name}.txt', 'w') as f:
        f.write("--- Ensemble: Promedio de probabilidades (umbral 0.5) ---\n")
        f.write(classification_report(y_true, y_pred_binary, target_names=classes))
        f.write("\nMatriz de confusión:\n")
        f.write(str(confusion_matrix(y_true, y_pred_binary)))
        f.write("\n\n--- Ensemble: Votación mayoritaria ---\n")
        f.write(classification_report(y_true, y_majority, target_names=classes))
        f.write("\nMatriz de confusión:\n")
        f.write(str(confusion_matrix(y_true, y_majority)))
    print(f"¡Reporte guardado como ensemble_classification_report_{dataset_name}.txt!")

print("\n==== Evaluación ensemble terminada para los 4 sets ====")




