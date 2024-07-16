#project_5
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models, regularizers, optimizers
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

# Load and preprocess validation images
X_validation = []
y_validation = []
for image_path in validation_images:
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    X_validation.append(img_array)
    y_validation.append(1 if 'butterfly' in image_path else 0)  # Label 1 for butterfly, 0 for other

# Load and preprocess testing images
X_test = []
y_test = []
for image_path in test_images:
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    X_test.append(img_array)
    y_test.append(1 if 'butterfly' in image_path else 0)  # Label 1 for butterfly, 0 for other

# Convert lists to numpy arrays
X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(y_train)
X_validation = tf.convert_to_tensor(X_validation)
y_validation = tf.convert_to_tensor(y_validation)
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)

# Define optimization algorithms to compare
optimizers_to_compare = [
    ('SGD', optimizers.SGD()),
    ('Adam', optimizers.Adam()),
    ('RMSprop', optimizers.RMSprop()),
    ('Adagrad', optimizers.Adagrad())
]

# Compare performance with different optimizers
for optimizer_name, optimizer in optimizers_to_compare:
    print(f"\nTraining with {optimizer_name} optimizer...")

    # Model definition
    model = models.Sequential([
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

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Training
    history = model.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_validation, y_validation)
    )

    # Evaluation on training set
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    print(f'Training Accuracy with {optimizer_name} optimizer: {train_accuracy}')

    # Evaluation on validation set
    val_loss, val_accuracy = model.evaluate(X_validation, y_validation, verbose=0)
    print(f'Validation Accuracy with {optimizer_name} optimizer: {val_accuracy}')

    # Evaluation on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy with {optimizer_name} optimizer: {test_accuracy}')

    # Print final accuracies for each optimizer
    print(f"Final Test Accuracy with {optimizer_name} optimizer: {test_accuracy}")
