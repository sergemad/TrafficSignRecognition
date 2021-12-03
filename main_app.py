#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

model = load_model('model.h5')

# Storing all lables
all_lables = ['Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)',
              'Speed limit (70km/h)','Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)',
              'Speed limit (120km/h)','No passing','No passing for vechiles over 3.5 metric tons',
              'Right-of-way at the next intersection','Priority road','Yield','Stop','No vechiles',
              'Vechiles over 3.5 metric tons prohibited','No entry','General caution','Dangerous curve to the left',
              'Dangerous curve to the right','Double curve','Bumpy road','Slippery road','Road narrows on the right',
              'Road work','Traffic signals','Pedestrians','Children crossing','Bicycles crossing','Beware of ice/snow',
              'Wild animals crossing','End of all speed and passing limits','Turn right ahead','Turn left ahead',
              'Ahead only','Go straight or right','Go straight or left','Keep right','Keep left','Roundabout mandatory',
              'End of no passing','End of no passing by vechiles over 3.5 metric']

#Setting Title of App
st.title("Traffic sign prediction")
st.markdown("Upload an image of a traffic sign")

#Uploading the dog image
image = st.file_uploader("Choose an image...")
submit = st.button('Predict')

#On predict button click
if submit:


    if image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (50,50))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,50,50,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)

        st.title(str("The Traffic Sign is "+all_lables[np.argmax(Y_pred)]))