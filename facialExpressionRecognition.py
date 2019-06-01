from keras.models import load_model
from PIL import Image
import cv2
import numpy as np

#Defining labels 
class facialExpressionRecognition(object):
    def __init__(self):
         #Loading pre-trained model
        self.pretrained_model = load_model("Resources/fer2013_mini_XCEPTION.99-0.65.hdf5")

    def get_label(self, argument):
        labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad' , 5:'Surprise', 6:'Neutral'}
        return(labels.get(argument, "Invalid emotion"))

    # Refactor above steps into reusable function
    def model_predict(self, image_path):
        
        img = Image.open(image_path)
        face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(np.asarray(img), 1.3, 5)
        if len(faces) < 1:return "No Face Detected"
        for (x, y, w, h) in faces:
            if len(faces) == 1: #Use simple check if one face is detected, or multiple (measurement error unless multiple persons on image)
                crop_img = img.crop((x,y,x+w,y+h))
            else:return "multiple face detected"
        ##display(crop_img)
        
       
        
        #Resizing image to required size
        test_image = crop_img.resize((64,64),Image.ANTIALIAS)

        #Converting image to array
        test_image = np.array(test_image)

        #converting to grayscale
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        #scale pixels values to lie between 0 and 1 because we did same to our train and test set
        gray = gray/255

        #reshaping image (-1 is used to automatically fit an integer at it's place to match dimension of original image)
        gray = gray.reshape(-1, 64, 64, 1)

        res = self.pretrained_model.predict(gray)

        #argmax returns index of max value
        result_num = np.argmax(res[0])

        #get emotion name
        emotion = self.get_label(result_num)

        # print predictions
        print("\nProbabilities are " + str(res[0])+"\n")
        print("Emotion is "+ self.get_label(result_num))

        return emotion