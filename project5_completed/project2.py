##project_2 without using function
import os
import numpy as np
import tensorflow as tf

train_dir = '/home/afreen-mohammad/Downloads/archive/training/'
validation_dir = '/home/afreen-mohammad/Downloads/archive/validation/'
test_dir = '/home/afreen-mohammad/Downloads/archive/testing/'

train_filenames = []
train_labels = []
label_index = 0
for filename in os.listdir(train_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        train_filenames.append(os.path.join(train_dir, filename))
        train_labels.append(label_index)

validation_filenames = []
validation_labels = []
label_index = 0
for filename in os.listdir(validation_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        validation_filenames.append(os.path.join(validation_dir, filename))
        validation_labels.append(label_index)

test_filenames = []
test_labels = []
label_index = 0
for filename in os.listdir(test_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        test_filenames.append(os.path.join(test_dir, filename))
        test_labels.append(label_index)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

batch_size = 32
steps_per_epoch_train = len(train_filenames) // batch_size
steps_per_epoch_validation = len(validation_filenames) // batch_size

train_generator = train_datagen.flow(
    np.array([tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(filename, target_size=(28, 28))) for filename in train_filenames]),
    np.array(train_labels),
    batch_size=batch_size,
    shuffle=True
)

validation_generator = validation_datagen.flow(
    np.array([tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(filename, target_size=(28, 28))) for filename in validation_filenames]),
    np.array(validation_labels),
    batch_size=batch_size,
    shuffle=True
)

test_generator = test_datagen.flow(
    np.array([tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(filename, target_size=(28, 28))) for filename in test_filenames]),
    np.array(test_labels),
    batch_size=batch_size,
    shuffle=False
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Change the number of epochs to 20
epochs = 20

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_train,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation
)

training_accuracy = history.history['sparse_categorical_accuracy']
validation_accuracy = history.history['val_sparse_categorical_accuracy']

history.history['training accuracy'] = history.history.pop('sparse_categorical_accuracy')
history.history['validation accuracy'] = history.history.pop('val_sparse_categorical_accuracy')

test_loss, test_accuracy = model.evaluate(
    test_generator,
    steps=len(test_filenames) // batch_size
)

print(f'Training accuracy: {training_accuracy[-1]}')
print(f'Validation accuracy: {validation_accuracy[-1]}')
print(f'Test accuracy: {test_accuracy}')
