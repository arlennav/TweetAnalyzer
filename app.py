import os
import json

import pandas as pd
import numpy as np

from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine,func
from flask_sqlalchemy import SQLAlchemy

from flask import Flask, jsonify, render_template,redirect, url_for,request,flash
from werkzeug.utils import secure_filename
from os.path import join, dirname, realpath
from TwitterAPI import TwitterAPI


from config import *
from cnnCifar100 import CIFAR100model
from xceptionClassification import xceptionClassification
from sentimentAnalysis import sentimentAnalysis
from facialExpressionRecognition import facialExpressionRecognition

import matplotlib.pyplot as plt
from PIL import Image
from resizeimage import resizeimage

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#################################################
# Database Setup
#################################################
app.config['SQLALCHEMY_DATABASE_URI'] = mysqlcs

db = SQLAlchemy(app)
# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(db.engine, reflect=True)
# Save references to each table
Tweets = Base.classes.tweets

'''
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
'''

def init():
    global sentiment_model,xception_model,cnn_cifar100_model,facial_emotion_model
    #Sentiment analysis
    sentiment_model = sentimentAnalysis()
    #Xception
    xception_model= xceptionClassification()
    #Cifar100
    cnn_cifar100_model= CIFAR100model()
    cnn_cifar100_model.load_model()
    #Facial Emotion
    facial_emotion_model = facialExpressionRecognition()
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

'''
send tweet
'''
@app.route('/api/tweet/<int:tweetID>', methods=['POST'])
def tweet(tweetID):
    try:
        result = db.session.query(Tweets).filter(Tweets.id==int(tweetID)).first()
        if result:
            api = TwitterAPI(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)
            with open(f'static/img/upload/{result.imagename}', 'rb') as file:
                data = file.read()
                tweet=f'This is a {result.tweetsentiment} tweet. \n\nCifar100: {result.imagetypeCifar} \n' 
                tweet+=f'Xception: {result.imagetypeXception} \n\n'      
                tweet+=f'{result.tweet}'
                r = api.request('statuses/update_with_media', {'status': tweet}, {'media[]':data})
                return jsonify(data=r.status_code)
    except e:
        print(e)

@app.route("/",methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
        valtweet=request.form['tweet']
        if not valtweet:
            flash('Please include your tweet.')
            return redirect(request.url)
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No attachemnt.')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save uploaded image
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            uploaded_img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            filename_original, file_extension = os.path.splitext(filename) 
            image = Image.open(uploaded_img_path)
            thumb_image = resizeimage.resize_contain(image, [200, 200])
            thumb_image.save(os.path.join(app.config['UPLOAD_FOLDER'], f'{filename_original}_thumb{file_extension}'))
           
            #Preditct image category
            #Cifar
            uploaded_img = plt.imread(uploaded_img_path)
            imagetypeCifar=cnn_cifar100_model.model_predict(uploaded_img)
            if imagetypeCifar=='':
                imagetypeCifar='Not detected'
            
            #Xception
            imagetypeXception=xception_model.model_predict(uploaded_img_path)
            if imagetypeXception=='':
                imagetypeXception='Not detected'
                
            #SentimentAnalysis
            sentiment_result=sentiment_model.model_predict(valtweet)

            #Facial Emotion
            facial_emotion_result=facial_emotion_model.model_predict(uploaded_img_path)
            if facial_emotion_result=='':
                facialEmotion='Not detected'
            
            #Insert into Tweets table
            tweet = Tweets(tweet=valtweet, tweetsentiment=sentiment_result['label'], 
                           imagename=filename,imagetypeCifar=imagetypeCifar.upper(),
                           imagetypeXception=imagetypeXception.upper(),facialEmotion = facial_emotion_result
                          )
            db.session.add(tweet)
            db.session.commit()
            
            flash('Record was successfully added.')
            return redirect(request.url)
    else:
        #Query table
        results = db.session.query(Tweets).order_by(Tweets.id.desc()).all()
        # Create a dictionary from the row data and append to a list of all_passengers
        all_tweets = []
        for tweet in results:
            tweet_dict = {}
            tweet_dict["id"] = tweet.id
            tweet_dict["tweet"] = tweet.tweet
            tweet_dict["tweetsentiment"] = tweet.tweetsentiment
            tweet_dict["imagename"] = tweet.imagename
            filename_original, file_extension = os.path.splitext(tweet.imagename)
            tweet_dict["imagename_thumb"] = f'{filename_original}_thumb{file_extension}'
            tweet_dict["imagetypeCifar"] = tweet.imagetypeCifar
            tweet_dict["imagetypeXception"] = tweet.imagetypeXception
            tweet_dict["facialEmotion"] = tweet.facialEmotion
            all_tweets.append(tweet_dict)
        return render_template("index.html",tweets=all_tweets)

if __name__ == "__main__":
    print(("Loading Keras model and Flask starting server, please wait until server has fully started..."))
    init()
    app.run(threaded = False)
