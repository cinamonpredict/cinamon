import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("saved_model/gradenewone.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a image file")

map_dict = { 0:'alba',
             1:'blackpeper_low',
             2:'c4',
             3:'c4_c5',
             4:'extraspecial',
             5:'faq',
             6:'grade_1',
             7:'grade_2',
             8:'NoGrade'
            }


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        # if(map_dict [prediction] == 'turmericfingers'){
        #     Grade = 1
        # }
        st.text(map_dict [prediction])
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
        if map_dict [prediction] == "alba":
            st.text_area("Market Price per 1kg:4900-5000LKR")
        if map_dict [prediction] == "blackpeper_low":
            st.text_area("Market Price per 1kg:2000LKR")
        if map_dict [prediction] == "c4":
            st.text_area("Market Price per 1kg:3800-4000LKR")
        if map_dict [prediction] == "c4_c5":
             st.text_area("Market Price per 1kg:3800-4000LKR")
        if map_dict [prediction] == "extraspecial": 
             st.text_area("Market Price per 1kg:4500-5000LKR")
        if map_dict [prediction] == "faq": 
             st.text_area("Market Price per 1kg:2500-3000LKR")
        if map_dict [prediction] == "grade_1": 
             st.text_area("Market Price per 1kg:2300-2500LKR")
        if map_dict [prediction] == "grade_2": 
             st.text_area("Market Price per 1kg:2000-2200LKR")
        if map_dict [prediction] == "NoGrade": 
             st.text_area("This Is not a Spice")
        
