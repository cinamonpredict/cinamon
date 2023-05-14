import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("saved_model/md.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a image file")

map_dict = {0:'StripeCanker',
            1:'RoughBark',
            2:'Sudupulli-White-spot',
            3:'Leaf-Blight',
            4:'Gammiris-pala-makka',
            5:'Diconocris-Distani-drake',
            6:'NoDisease'
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
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
        if map_dict [prediction] == "Sudupulli-White-spot":
            st.text_area("If this disease heavily spreads in your field, apply insecticides which include Imidacloprid. 10ml per 10liter of water")
        if map_dict [prediction] == "StripeCanker":
            st.text_area("maintenance of good soil drainage• Maintain proper sunlight")
        if map_dict [prediction] == "RoughBark":
            st.text_area("Apply fertilizer for Correct time and standard.maintenance of good soil drainage")
        if map_dict [prediction] == "Leaf-Blight":
            st.text_area("Control Shading.Protect water capacity of the soil by using soil covers.Remove effected branches and leaves.Manage suitable fertilizer.")
        if map_dict [prediction] == "Gammiris-pala-makka": 
            st.text_area("Remove effected leaves.Use insecticide which include dimethoate chemical (25ml per 10L of water)")
        if map_dict [prediction] == "Diconocris-Distani-drake": 
            st.text_area("mospilan insecticide (25ml per 10L of water)")
                        
                        
        
