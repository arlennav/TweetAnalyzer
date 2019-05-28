# Dependencies
import matplotlib.pyplot as plt

import os
import numpy as np
import tensorflow as tf


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import (
    Xception, 
    preprocess_input, 
    decode_predictions
)


class xceptionClassification(object):

    def __init__(self):
        #this model gets to a top-1 validation accuracy of 0.790   
        self.model  = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    
    def model_predict(self,image_path):
        img = image.load_img(image_path, target_size=(299,299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = self.model.predict(x)
        pred=decode_predictions(predictions, top=3)
        return ','.join([item[1] for item in pred[0]])