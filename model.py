import csv
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Lambda, Cropping2D


# Parameters
EPOCHS = 5
BATCH_SIZE = 64
STEERING_CORRECTION = 0.2
VALIDATION_SIZE = 0.2
LIMIT_SAMPLES = 0  # Set to low number for quick testing (only processes first x lines of CSV)


# Variables
lines = []
images = []
values = []


# Load CSV file
with open('data/driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)

# Limit number of samples
# ([1:] to skip header)
if LIMIT_SAMPLES > 0:
    lines = lines[1:LIMIT_SAMPLES]
else:
    lines = lines[1:]

# Split data
train_samples, validation_samples = train_test_split(lines, test_size=VALIDATION_SIZE)

# Lengths (times 3 for center/left/right, times 2 for mirroring)
train_samples_len = len(train_samples) * 3 * 2
validation_samples_len = len(validation_samples) * 3 * 2
print('Training samples: ', train_samples_len)
print('Validation samples: ', validation_samples_len)


# Generate training or validation data on-the-fly
def generator(samples):
    no_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, no_samples, BATCH_SIZE):
            batch = samples[offset:offset + BATCH_SIZE]
            images = []
            values = []
            for batch_sample in batch:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = 'data/IMG/' + filename
                    image = cv2.imread(current_path)[:,:,::-1]  # Read image and convert from BGR to RGB
                    images.append(image)

                    # Adjust steering angle for left/right images
                    if i == 0:
                        value = float(batch_sample[3])
                    elif i == 1:
                        value = float(batch_sample[3]) + STEERING_CORRECTION
                    elif i == 2:
                        value = float(batch_sample[3]) - STEERING_CORRECTION
                    values.append(value)

                    # Augment data with mirrored version of each image
                    images.append(cv2.flip(image, 1))
                    values.append(value * -1.0)

            X_train = np.array(images)
            y_train = np.array(values)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)


# Model
model = Sequential()

# Normalizing, mean centering, and cropping
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (1, 1))))

# Actual model (nVidia)
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Mean squared error as loss function, Adam as optimizer
model.compile(loss='mse', optimizer='adam')

# Train
history = model.fit_generator(train_generator, samples_per_epoch=train_samples_len, validation_data=validation_generator, nb_val_samples=validation_samples_len, nb_epoch=EPOCHS, verbose=1)

model.save('model.h5')

# Output visualization
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model MSE loss')
plt.ylabel('MSE loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()
