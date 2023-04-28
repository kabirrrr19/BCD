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

#create a dataframe to run the predictions
test_df = pd.DataFrame({'id':os.listdir(test_path)})
# print(test_df.head())
datagen_test = ImageDataGenerator(rescale=1./255.)
test_generator = datagen_test.flow_from_dataframe(
    dataframe=test_df,
    directory=test_path,
    x_col='id', 
    y_col=None,
    target_size=(64,64),         # original image = (96, 96) 
    batch_size=1,
    shuffle=False,
    class_mode=None)

model = tf.keras.models.load_model("my_model")
predictions = model.predict(test_generator, verbose=1)

# print(len(predictions))

#create submission dataframe
predictions = np.transpose(predictions)[0]
submission_df = pd.DataFrame()
submission_df['id'] = test_df['id'].apply(lambda x: x.split('.')[0])
submission_df['label'] = list(map(lambda x: 0 if x < 0.5 else 1, predictions))
submission_df.head()
submission_df['label'].value_counts()

#plot test predictions
sns.countplot(data=submission_df, x='label').set(title='Predicted Labels for Test Set')
plt.show()

#convert to csv to submit to competition
submission_df.to_csv('submission.csv', index=False)