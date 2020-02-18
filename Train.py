import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import os
from os import listdir
from os.path import isdir
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pickle
import time

NAME = 'Face Detection-{}'.format(int(time.time()))
tensorBoard = TensorBoard(log_dir = 'logs\\-{}'.format(NAME))
DIR = 'Dataset'
IMG_SIZE = 130
training_data = []
X = []
y = []

def get_lables(directory):
  labels = list()
  for subdir in listdir(directory):
	  path = subdir
	  labels.append(path)
  return labels

labels = get_lables(DIR)

def create_training_data():
  for label in labels:
    path = os.path.join(DIR,label)
    class_num = labels.index(label)
    for img in listdir(path):
      try:
        img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
        img_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([img_array, class_num])
      except Exception as e:
        pass

create_training_data()

for feature, label in training_data:
  X.append(feature)
  y.append(label)
 
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.asarray(y)
pickle_out = open('X.pickle','wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle','wb')
pickle.dump(y, pickle_out)
pickle_out.close()

X = pickle.load(open('X.pickle','rb'))
y = pickle.load(open('y.pickle','rb'))

X = X/255.0

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("sigmoid"))

model.add(Dense(3))
model.add(Activation("sigmoid"))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs = 5, validation_split = 0.1, callbacks = [tensorBoard])

model.save('Face_Detection.h5')