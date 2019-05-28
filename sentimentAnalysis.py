# Dependencies
import os
import numpy as np
import tensorflow as tf
from config import *

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle

class sentimentAnalysis(object):

    def __init__(self):
        # load the pre-trained Keras model for sentiment analysis
        self.model = load_model('resources/sentiment_model.h5')
        self.tokenizer = Tokenizer()
        with open('resources/tokenizer.pkl', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
    
    def decode_sentiment(self,score, include_neutral=True):
        if include_neutral:        
            label = NEUTRAL
            if score <= SENTIMENT_THRESHOLDS[0]:
                label = NEGATIVE
            elif score >= SENTIMENT_THRESHOLDS[1]:
                label = POSITIVE
            return label
        else:
            return NEGATIVE if score < 0.5 else POSITIVE
        
    def model_predict(self,text, include_neutral=True):
        # Tokenize text
        x_test = pad_sequences(self.tokenizer.texts_to_sequences([text]), maxlen=300)
        # Predict
        score = self.model.predict([x_test])[0]
        # Decode sentiment
        label = self.decode_sentiment(score, include_neutral=include_neutral)
        return {"label": label, "score": float(score)}  
