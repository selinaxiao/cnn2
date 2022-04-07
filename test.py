import glob
import Code
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split


output = glob.glob("arcDataset/*/*")
#print(output)
print(len(output))

t_lower = 300
t_upper = 350

def show_image(image,title):
    plt.imshow(image,cmap=cm.gray)
    plt.title(title)
    plt.show()

result = []


image_input = []
for filename in output:
    img = cv2.imread(filename)
    image_input.append(img)
    out = cv2.Canny(img, t_lower, t_upper)
    #show_image(out, 'Final Image for'+filename)
    result.append(out)

print(len(result))


import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

data_train, data_test, labels_train, labels_test = train_test_split(image_input, result, test_size=0.20, random_state=42)

show_image(data_train[0], 'Original Image')
show_image(labels_train[0], 'Final Image')