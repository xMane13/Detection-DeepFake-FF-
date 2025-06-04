import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Información de cada modelo y su test set correspondiente
model_test_pairs = [
    ('xception_ffpp_best.h5', '/workspace/FFPP_data/faces_split'),
    ('xception_face2face_best.h5', '/workspace/FFPP_data/faces_split_face2face'),
    ('xception_deepfakes_best.h5', '/workspace/FFPP_data/faces_split_deepfakes'),
    ('xception_neuraltextures_best.h5', '/workspace/FFPP_data/faces_split_neuraltextures'),
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

for model_file, data_dir in model_test_pairs:
    print(f"\nEvaluando modelo: {model_file}")
    model = load_model(model_file)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        data_dir + '/test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    target_names = list(test_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=target_names)
    cm = confusion_matrix(y_true, y_pred)

    print("CLASSES:", target_names)
    print(report)
    print("Matriz de confusión:\n", cm)

    # Guarda el reporte a un archivo único para cada modelo
    out_name = model_file.replace('.h5', '_classification_report.txt')
    with open(out_name, 'w') as f:
        f.write("CLASSES: " + str(target_names) + "\n\n")
        f.write(report)
        f.write("\nMatriz de confusión:\n")
        f.write(str(cm))
    print(f"¡Reporte guardado como {out_name}!")

print("\n¡Evaluación terminada para todos los modelos!")


