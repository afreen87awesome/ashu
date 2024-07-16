##project_1
import os
import numpy as np
from PIL import Image

# Directories
train_dir = '/home/afreen-mohammad/Downloads/archive/training/'
validation_dir = '/home/afreen-mohammad/Downloads/archive/validation/'
test_dir = '/home/afreen-mohammad/Downloads/archive/testing/'

# Load training data
train_images = []
train_labels = []
for file in os.listdir(train_dir):
    img = Image.open(os.path.join(train_dir, file))
    img = img.resize((256, 256))
    img = img.convert('L')
    train_images.append(np.array(img))
    try:
        label = int(file.split('_')[0])
        if label >= 0 and label <= 9:
            train_labels.append(label)
        else:
            train_labels.append(0)  # or any other default value
    except ValueError:
        train_labels.append(0)  # or any other default value

train_images = np.array(train_images).reshape((-1, 256*256))
train_labels = np.array(train_labels)

# Load validation data
validation_images = []
validation_labels = []
for file in os.listdir(validation_dir):
    img = Image.open(os.path.join(validation_dir, file))
    img = img.resize((256, 256))
    img = img.convert('L')
    validation_images.append(np.array(img))
    try:
        label = int(file.split('_')[0])
        if label >= 0 and label <= 9:
            validation_labels.append(label)
        else:
            validation_labels.append(0)  # or any other default value
    except ValueError:
        validation_labels.append(0)  # or any other default value

validation_images = np.array(validation_images).reshape((-1, 256*256))
validation_labels = np.array(validation_labels)

# Load test data
test_images = []
test_labels = []
for file in os.listdir(test_dir):
    img = Image.open(os.path.join(test_dir, file))
    img = img.resize((256, 256))
    img = img.convert('L')
    test_images.append(np.array(img))
    try:
        label = int(file.split('_')[0])
        if label >= 0 and label <= 9:
            test_labels.append(label)
        else:
            test_labels.append(0)  # or any other default value
    except ValueError:
        test_labels.append(0)  # or any other default value

test_images = np.array(test_images).reshape((-1, 256*256))
test_labels = np.array(test_labels)

# Perceptron
learning_rate = 0.01
n_iters = 200  # Changed number of iterations to 200
weights = np.zeros(256*256)
bias = 0

for epoch in range(n_iters):
    # Train the perceptron
    for idx, x_i in enumerate(train_images):
        predicted = np.dot(x_i, weights) + bias
        predicted = np.where(predicted >= 0, 1, 0)
        update = learning_rate * (train_labels[idx] - predicted)
        weights += update * x_i
        bias += update
    
    # Evaluate on training set
    train_predictions = np.dot(train_images, weights) + bias
    train_predictions = np.where(train_predictions >= 0, 1, 0)
    train_accuracy = np.mean(train_predictions == train_labels)

    # Evaluate on validation set
    val_predictions = np.dot(validation_images, weights) + bias
    val_predictions = np.where(val_predictions >= 0, 1, 0)
    val_accuracy = np.mean(val_predictions == validation_labels)
    
    # Evaluate on test set
    test_predictions = np.dot(test_images, weights) + bias
    test_predictions = np.where(test_predictions >= 0, 1, 0)
    test_accuracy = np.mean(test_predictions == test_labels)

    # Print accuracies for each epoch
    print(f'Epoch {epoch+1}/{n_iters}')
    print(f'Training accuracy: {train_accuracy:.4f}')
    print(f'Validation accuracy: {val_accuracy:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')


