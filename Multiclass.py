import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

IMG_SIZE = (224,224)
BATCH_SIZE = 32
DATA_DIR = '/home/manuel.munoz__yachaytech.edu.ec/FFPP_data/faces_multiclass_split'
EPOCHS = 20

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR + '/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_test_datagen.flow_from_directory(
    DATA_DIR + '/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_gen = val_test_datagen.flow_from_directory(
    DATA_DIR + '/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

base_model = Xception(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('xception_multiclass_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
early = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
csv_logger = CSVLogger('training_log_multiclass.csv')

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint, early, csv_logger]
)

# Graficar curvas
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Multiclass Xception')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Multiclass Xception')

plt.tight_layout()
plt.savefig('training_curves_multiclass.png')
plt.close()

# Evaluación en test y classification report
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

target_names = list(test_gen.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=target_names)
cm = confusion_matrix(y_true, y_pred)

print(report)
print("Matriz de confusión:\n", cm)
print("Accuracy:", accuracy_score(y_true, y_pred))

with open('classification_report_multiclass.txt', 'w') as f:
    f.write(report)
    f.write("\nMatriz de confusión:\n")
    f.write(str(cm))
    f.write("\nAccuracy: %.4f\n" % accuracy_score(y_true, y_pred))
print("¡Entrenamiento y evaluación multiclase completados!")

