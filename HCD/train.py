import numpy as np 
import pandas as pd 
import os
import random
from sklearn.utils import shuffle
import shutil

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.patches as patches

# Work with images
# from skimage.transform import rotate
# import skimage.io as io
import cv2 as cv
from PIL import Image

# Model Development
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomZoom, RandomRotation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)



# Get files
test_path = '../histopathologic-cancer-detection/test/'
train_path = '../histopathologic-cancer-detection/train/'
sample_submission = pd.read_csv('../histopathologic-cancer-detection/sample_submission.csv')
train_data = pd.read_csv('../histopathologic-cancer-detection/train_labels.csv')
# print(train_data)


# declare constants for reproduciblity
RANDOM_STATE = 49

# have a look at the format of the data
# print(train_data.head())

# print(len(os.listdir('../histopathologic-cancer-detection/train')))
# print(len(os.listdir('../histopathologic-cancer-detection/test')))

# take a look at the data further
# print(train_data.describe())

# check information, data types, and for missing data
# print(train_data.info())

# print(pd.DataFrame(data={'Label Counts': train_data['label'].value_counts()}))
# sns.countplot(x=train_data['label'], palette='colorblind').set(title='Label Counts Histogram')
# plt.show()

# global
#create pie chart
# fig = px.pie(train_data, 
#              values = train_data['label'].value_counts().values, 
#              names = train_data['label'].unique())
# fig.update_layout(
#     title={
#         'text': "Label Percentage Pie Chart",
#         'y':.99,
#         'x':0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'})
# fig.show()


# Visualize a few images
# fig, ax = plt.subplots(5, 5, figsize=(15, 15))
# for i, axis in enumerate(ax.flat):
#     file = str(train_path + train_data.id[i] + '.tif')
#     image = Image.open(file)
#     axis.imshow(image)
#     box = patches.Rectangle((32,32),32,32, linewidth=2, edgecolor='r',facecolor='none', linestyle='-')
#     axis.add_patch(box)
#     axis.set(xticks=[], yticks=[], xlabel = train_data.label[i]);
#     cv.waitKey(0)
    # image.show()
    # plt.show()

BATCH_SIZE = 256

# prepare data for training
def append_tif(string):
    return string+".tif"

train_data["id"] = train_data["id"].apply(append_tif)
train_data['label'] = train_data['label'].astype(str)

# randomly shuffle training data
train_data = shuffle(train_data, random_state=RANDOM_STATE)

# modify training data by normalizing it and split data into training and validation sets

datagen = ImageDataGenerator(rescale=1./255.,
                            validation_split=0.15)

# generate training data
train_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=train_path,
    x_col="id",
    y_col="label",
    subset="training",
    batch_size=BATCH_SIZE,
    seed=RANDOM_STATE,
    class_mode="binary",
    target_size=(64,64))        # original image = (96, 96) 


# generate validation data
valid_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=train_path,
    x_col="id",
    y_col="label",
    subset="validation",
    batch_size=BATCH_SIZE,
    seed=RANDOM_STATE,
    class_mode="binary",
    target_size=(64,64))       # original image = (96, 96) 


# Setup GPU accelerator - configure Strategy. Assume TPU...if not set default for GPU/CPU
tpu = None
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy()


# set ROC AUC as metric

ROC_1 = tf.keras.metrics.AUC()

# use GPU
with strategy.scope():kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3


model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (64, 64, 3))) # original image = (96, 96, 3) 
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

#compile
adam_optimizer = Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', metrics=['accuracy', ROC_1], optimizer=adam_optimizer)


EPOCHS = 20

# train the model
history_model = model.fit(
                        train_generator,
                        epochs = EPOCHS,
                        validation_data = valid_generator)

# get the metric names so we can use evaulate_generator
# print(model.metrics_names)
model.save('./myModel.h5')

# # plot model accuracy per epoch 
# plt.plot(history_model.history['accuracy'])
# plt.plot(history_model.history['val_accuracy'])
# plt.title('Model One Accuracy per Epoch')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validate'], loc='upper left')
# plt.show()

# # plot model loss per epoch
# plt.plot(history_model.history['loss'])
# plt.plot(history_model.history['val_loss'])
# plt.title('Model One Loss per Epoch')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validate'], loc='upper left')
# plt.show()


# # plot model ROC per epoch
# plt.plot(history_model.history['auc_1'])
# plt.plot(history_model.history['val_auc_1'])
# plt.title('Model One AUC ROC per Epoch')
# plt.ylabel('ROC')
# plt.xlabel('epoch')
# plt.legend(['train', 'validate'], loc='upper left')
# plt.show()


