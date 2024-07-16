#####project_4

##ptoject_4 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models, regularizers
import os

# Define paths
train_dir = '/home/afreen-mohammad/Downloads/archive/training'
validation_dir = '/home/afreen-mohammad/Downloads/archive/validation'
test_dir = '/home/afreen-mohammad/Downloads/archive/testing'

# Function to get list of image filenames from directory
train_images = []
validation_images = []
test_images = []

# Populate image lists
for filename in os.listdir(train_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        train_images.append(os.path.join(train_dir, filename))

for filename in os.listdir(validation_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        validation_images.append(os.path.join(validation_dir, filename))

for filename in os.listdir(test_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        test_images.append(os.path.join(test_dir, filename))

# Number of images
num_train_images = len(train_images)
num_validation_images = len(validation_images)
num_test_images = len(test_images)

# Print number of images for verification
print(f"Number of training images: {num_train_images}")
print(f"Number of validation images: {num_validation_images}")
print(f"Number of testing images: {num_test_images}")

# Image data preprocessing and augmentation directly in the loop
X_train = []
y_train = []
for image_path in train_images:
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0  # Rescale pixel values to [0, 1]
    X_train.append(img_array)
    y_train.append(1 if 'butterfly' in image_path else 0)  # Label 1 for butterfly, 0 for other

X_validation = []
y_validation = []
for image_path in validation_images:
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    X_validation.append(img_array)
    y_validation.append(1 if 'butterfly' in image_path else 0)

X_test = []
y_test = []
for image_path in test_images:
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    X_test.append(img_array)
    y_test.append(1 if 'butterfly' in image_path else 0)

# Convert lists to numpy arrays
X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(y_train)
X_validation = tf.convert_to_tensor(X_validation)
y_validation = tf.convert_to_tensor(y_validation)
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)

# Model with L1 regularization
model_l1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_l1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training L1 Regularization Model...")
history_l1 = model_l1.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_validation, y_validation)
)

# Print training accuracy, validation accuracy, and testing accuracy for L1
train_acc_l1 = history_l1.history['accuracy'][-1]
val_acc_l1 = history_l1.history['val_accuracy'][-1]
loss_l1, accuracy_l1 = model_l1.evaluate(X_test, y_test)
print(f'L1 Regularization Training Accuracy: {train_acc_l1}')
print(f'L1 Regularization Validation Accuracy: {val_acc_l1}')
print(f'L1 Regularization Test Accuracy: {accuracy_l1}')

# Model with L2 regularization
model_l2 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_l2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining L2 Regularization Model...")
history_l2 = model_l2.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_validation, y_validation)
)

# Print training accuracy, validation accuracy, and testing accuracy for L2
train_acc_l2 = history_l2.history['accuracy'][-1]
val_acc_l2 = history_l2.history['val_accuracy'][-1]
loss_l2, accuracy_l2 = model_l2.evaluate(X_test, y_test)
print(f'L2 Regularization Training Accuracy: {train_acc_l2}')
print(f'L2 Regularization Validation Accuracy: {val_acc_l2}')
print(f'L2 Regularization Test Accuracy: {accuracy_l2}')

# Model with Dropout regularization
model_dropout = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining Dropout Regularization Model...")
history_dropout = model_dropout.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_validation, y_validation)
)

# Print training accuracy, validation accuracy, and testing accuracy for Dropout
train_acc_dropout = history_dropout.history['accuracy'][-1]
val_acc_dropout = history_dropout.history['val_accuracy'][-1]
loss_dropout, accuracy_dropout = model_dropout.evaluate(X_test, y_test)
print(f'Dropout Regularization Training Accuracy: {train_acc_dropout}')
print(f'Dropout Regularization Validation Accuracy: {val_acc_dropout}')
print(f'Dropout Regularization Test Accuracy: {accuracy_dropout}')

# Print final accuracies for all techniques
print("\nFinal Test Accuracies:")
print(f"L1 Regularization Test Accuracy: {accuracy_l1}")
print(f"L2 Regularization Test Accuracy: {accuracy_l2}")
print(f"Dropout Regularization Test Accuracy: {accuracy_dropout}")
