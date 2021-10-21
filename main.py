#dependencies

import matplotlib
import cv2
from scipy.stats.mstats import linregress
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from impressionnet_multiview_final import ImpressionNet
from multiview_loss import multiview_loss

matplotlib.use('Agg')

%matplotlib inline
import matplotlib.pyplot as plt

#############################################
############## parameters ###################
#############################################

batch_size = 32
epoch_num = 75
learning_rate = 1e3
weight_decay = learning_rate/epoch_sum

#############################################
###### Data Augmentation for training #######

aug = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20, 
    zoom_range=0.15,
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.15,
    horizontal_flip=True, 
    fill_mode="nearest"
  )

#############################################
################ network ####################
#############################################

# load the trained model for trustworthiness score prediction trained on D1 dataset 
model_path = '/content/D1A.h5'
model = keras.models.load_model("D1A.h5", compile=False)
net = ImpressionNet.build()

##############################################
############### loss function ################

# loss function combination all losses, including Lmv, MSE loss for Regression
#input : Penultimate Dense Layers of FC namely "DenseLayer1" & "Denselayer2" 

##############################################

# retrieve the penultimate dense layers from model  
layer1 = net.get_layer("denselayer1")
layer2 = net.get_layer("denselayer2")

# construct the final loss function 
losses = {
    "model1_output": multiview_loss(layer1, layer2),
    "model2_output": multiview_loss(layer1, layer2)
}

##############################################

def main():

  img = cv2.imread('/content/test.png')
  img = cv2.resize(img, (224,224))
  x = image.img_to_array(img)
  plt.imshow(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  
  prediction = model.predict(np.expand_dims(x, axis=0))
  
  # Predictions of View 1 and View 2 along with the combined views
  
  print("View 1 prediction : ", prediction[0])
  print("View 2 prediction : ", prediction[1])
  print("Multi-view Prediction", (prediction[0]+prediction[1])/2)
 
main()
 
 
 
 
 
 
 
 
 
 
