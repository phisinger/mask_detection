# import seaborn as sns
from turtle import width
import streamlit as st
# import cv2
import os
import numpy as np
# import pickle
import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import models, utils
# import pandas as pd
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras import utils
import random
print("TensorFlow version:", tf.__version__)


# get model path
# parent_path = Path().parent.absolute()
model_path = os.path.join(os.getcwd(), "models", "final_face_detection_m.h5")

# load model
prediction_model = load_model(model_path)

# enable user to take a picture via webcam
img_file_buffer = st.camera_input("Take a picture")


def webcam_prediction(img_tensor=None):

    # preprocessing
    img_resized = tf.image.resize(img_tensor, [128, 128])
    img_array = np.array([img_resized])
    # print("resized image", img_resized.shape)

    # make prediction
    prediction = prediction_model.predict(img_array)
    print(prediction)
    return prediction


def stock_prediction():
    # select random image from stock collection
    stock_img_list = os.listdir(os.path.join(os.getcwd(), "test_imgs"))
    index = random.randint(0, len(stock_img_list)-1)
    img_path = os.path.join(os.getcwd(), "test_imgs", stock_img_list[index])
    # print(img_path)

    # Load image
    loaded_img = load_img(path=img_path,
                          color_mode="grayscale",
                          target_size=(128, 128))
    # display image
    st.image(Image.open(img_path), width=256)

    # preprocess image
    loaded_img = img_to_array(loaded_img)
    loaded_img = np.expand_dims(loaded_img, axis=0)

    # make prediction
    return prediction_model.predict(loaded_img)


if img_file_buffer is not None:
    # st.image(img_file_buffer)

    # To read image file buffer as a 3D uint8 tensor with TensorFlow:
    bytes_data = img_file_buffer.getvalue()
    img_tensor = tf.io.decode_image(bytes_data, channels=1, dtype=tf.float32)

    # Check the type of img_tensor:
    # Should output: <class 'tensorflow.python.framework.ops.EagerTensor'>
    st.write(type(img_tensor))

    # Check the shape of img_tensor:
    # Should output shape: (height, width, channels)
    st.write(img_tensor.shape)

    st.write("prediction")
    pred_result = webcam_prediction(img_tensor)
    if pred_result > 0.8:
        st.write("no mask")
    elif pred_result < 0.2:
        st.write("Mask!")
    else:
        st.write("Don't know")

if st.button("Predict from samples"):
    st.write("prediction")
    pred_result = stock_prediction()
    if pred_result > 0.8:
        st.write("no mask")
    elif pred_result < 0.2:
        st.write("Mask!")
    else:
        st.write("Don't know")
