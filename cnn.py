import numpy
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from PIL import Image

def show_image(image, title):
    plt.imshow(image, cmap=cm.gray)
    plt.title(title)
    plt.show()

target_height = 150
target_width = 200

arch_dataset_gs = glob.glob("arcDatasetGrayscale/*.jpg")
input_images = []
for filename in arch_dataset_gs:
    this_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    this_img = this_img.reshape((target_height, target_width, 1))
    input_images.append(this_img)

canny_gs = glob.glob("canny_imgs/*.jpg")
result = []
for filename in canny_gs:
    this_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    this_img = this_img.reshape(target_height*target_width)
    result.append(this_img)

image_train, image_test, labels_train, labels_test = train_test_split(input_images, result, test_size=0.20,
                                                                      random_state=42)
image_train = np.array(image_train)
labels_train = np.array(labels_train)

image_test = np.array(image_test)
labels_test = np.array(labels_test)

print(image_train.shape, labels_train.shape)
#show_image(image_train[0], 'Original Image')
#show_image(labels_train[0], 'Final Image')

# normalization_layer = layers.Rescaling(1. / 255)
# maxpool_layer = layers.MaxPooling2D((2, 2))
# labels_train=tf.convert_to_tensor(labels_train)
# maxpool_layer(labels_train)
# labels_train = np.array(labels_train)
# print(labels_train.shape)


# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image_train[i])
# plt.show()



model = models.Sequential()
# conv1,BN, Relu
model.add(layers.Conv2D(8, (3, 3), padding='same', input_shape=(target_height, target_width, 1)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

# conv2,BN, Relu,Pool
model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))

# conv3,BN, Relu,Pool
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))

# # conv4,BN, Relu
# model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
#
# # conv5,BN, Relu
# model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))


# FC, output 1
model.add(layers.Flatten())
model.add(layers.Dense(target_height*target_width))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])




history = model.fit(image_train, labels_train, epochs=5,
                    validation_data=(image_test, labels_test))
print(history)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(image_test, labels_test, verbose=2)

print(test_acc)
