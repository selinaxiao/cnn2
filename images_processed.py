import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split

arch_dataset = glob.glob("arcDataset/*/*.jpg")
grayscale_dataset = glob.glob('arcDatasetGrayscale/*.jpg')

t_lower = 300
t_upper = 350

def show_image(image, title):
    plt.imshow(image, cmap=cm.gray)
    plt.title(title)
    plt.show()

target_height = 150
target_width = 200

def resize_image(dataset, folder):
    for filename in dataset:
        img = cv2.imread(filename)
        # print(np.shape(img))
        img1 = tf.image.resize_with_crop_or_pad(img, target_height, target_width)
        this_img = tf.keras.utils.array_to_img(img1)
        idx = filename.rfind("\\")
        file_path = folder + '\\' + filename[idx+1::]
        this_img.save(file_path)
        print(file_path)

def make_grayscale(dataset, folder):
    for filename in dataset:
        img = cv2.imread(filename)
        # print(np.shape(img))
        img1 = tf.image.resize_with_crop_or_pad(img, target_height, target_width)
        img1 = tf.image.rgb_to_grayscale(img1, name=None)
        this_img = tf.keras.utils.array_to_img(img1)
        idx = filename.rfind("\\")
        file_path = folder + '\\' + filename[idx+1::]
        this_img.save(file_path)
        print(file_path)



def make_canny(dataset, folder):
    for filename in dataset:
        img = cv2.imread(filename)
        img1 = tf.image.resize_with_crop_or_pad(img, target_height, target_width)
        canny_img = cv2.Canny(img1.numpy(), t_lower, t_upper)
        canny_reshape = np.expand_dims(canny_img, 2)
        this_image = tf.keras.utils.array_to_img(canny_reshape)
        idx = filename.rfind("\\")
        file_path = folder + '\\' + filename[idx + 1::]
        this_image.save(file_path)





#resize_image(arch_dataset,'resized_images')
# make_grayscale(arch_dataset, 'arcDatasetGrayscale')
# make_canny(grayscale_dataset, 'canny_imgs')

