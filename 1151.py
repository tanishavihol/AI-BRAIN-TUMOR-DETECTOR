import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import keras
import tensorflow as tf
# from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.config.experimental_run_functions_eagerly(True)

img_dir = "Brain_Tumor_Dataset"

BATCH_SIZE = 64
IMAGE_SIZE = 150
input_shape = (150, 150, 1)

data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_data = data_gen.flow_from_directory(img_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, subset="training", color_mode="grayscale", shuffle=True, class_mode="binary")

# print(train_data)

val_data = data_gen.flow_from_directory(img_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, subset="validation", color_mode="grayscale", shuffle=False, class_mode="binary")

# print(val_data)

labels = train_data.class_indices
classes = list(labels.keys())
print(classes)


# MAKING THE MODEL
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization

model = Sequential()
model.add(keras.layers.InputLayer(input_shape=input_shape))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

with tf.device('/device:GPU:0'):
    history = model.fit(train_data, epochs=5, validation_data=val_data, verbose=1, steps_per_epoch=3681 // 64, validation_steps=919 // 64)

    # print("HISTORY", history)


# Evaluate the model
train_loss, train_acc = model.evaluate(train_data, steps=3681 // 64)
test_loss, test_acc = model.evaluate(val_data, steps=919 // 64)

print("TRAIN AND TEST LOSS AND ACCURACY", train_loss, test_loss, train_acc, test_acc)


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# save model
model.save("model.h5")


# testing model predicting good or bad
from matplotlib.pyplot import imshow
from PIL import Image, ImageOps

data = np.ndarray(shape=(1, 150, 150, 1), dtype=np.float32)
# image = Image.open("Brain_Tumor_Dataset/Brain_Tumor/Cancer (2100).jpg")
image = Image.open("Brain_Tumor_Dataset/Brain_Tumor/Cancer (1).jpg")
size = (150, 150)
image = ImageOps.grayscale(image)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
image_array = np.array(image)
# display(image)      # for jupiter notebook

# Display the image
plt.imshow(image_array, cmap='gray')  # Use grayscale colormap
plt.title("Input Image")
plt.axis('off')
plt.show()
# normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data = image_array.reshape((-1, 150, 150, 1))
# data[0] = normalized_image_array


prediction = model.predict(data)
# print("MODEL PREDICTION ===>", prediction[0][0])


if(prediction[0][0] == 1):
    print("PATIENT HAS NO BRAIN CANCER")
else:
    print("PATIENT HAS BRAIN CANCER")
