import numpy as np
from keras.utils import custom_object_scope
from model import TCNGenerator
import streamlit as st

@st.cache_resource
def load_tcn_model():
    with custom_object_scope({'TCNGenerator': TCNGenerator}):
        model = TCNGenerator(input_size=10)  
        dummy_input = np.random.random((1, 10, 4))
        model(dummy_input)
        model.load_weights('model/A4_B3_C2_Best.h5')  
    return model
