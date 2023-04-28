# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import numpy as np

# model = tf.keras.models.load_model('./myModel.h5')

# def preprocess(image):
#     img = Image.open(image)
#     img = img.resize((64, 64))
#     img = np.array(img.convert('L'))
#     img = img / 255.0
#     img = np.reshape(img, (1, 64, 64, 1))
#     return img

# def app():
#     st.title("Breast Cancer Detection")
#     st.write("Upload an image for Breast Cancer detection.")
#     uploaded_file = st.file_uploader("Choose an image...", type="jpg")
#     if uploaded_file is not None:
#         image = preprocess(uploaded_file)
#         prediction = model.predict(image)
#         if prediction > 0.5:
#             st.write("The image is predicted to have Breast Cancer.")
#         else:
#             st.write("The image is predicted to be Breast Cancer free.")


import streamlit as st
import tensorflow as tf
import numpy as np


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation = True)
def load_model():
    model = tf.keras.models.load_model(r"./myModel.h5")
    return model

model = load_model()

html_temp = """
    <div style="background-color:black;"><p style="color:white;font-size:40px;padding:9px">Malaria Detection Using Deep Learning</p></div>
    
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.sidebar.info("Artificial intelligence (AI) and open source tools, technologies, and frameworks are a powerful combination for improving society. 'Health is wealth' is perhaps a cliche, yet it's very accurate! We will use how AI can be leveraged for detecting the deadly disease malaria with a low-cost, effective, and accurate yet open source deep learning solution.")
# st.sidebar.text
    
st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
st.sidebar.info("KaustubhPandit24@gmail.com")

file = st.file_uploader('Upload the Cell Image', type=['jpg','png'])
# import cv2
from PIL import Image, ImageOps
def preprocess(image):
    img = image.resize((64, 64))
    # img = np.array(img.convert('L'))
    # img = img / 255.0
    # img = np.reshape(img, (1, 64, 64, 1))
    return img

def import_and_predict(image_data, model):
    prediction = model.predict(image_data)
    return prediction

if file is None:
    st.text("Upload the Cell Image here")
else:
    image = Image.open(file)
    image = preprocess(image)
    st.image(image, use_column_width = True)
    predictions = import_and_predict(image, model)
    if predictions > 0.5:
        string = "You have Malaria."
    else:
        string = "You Don't have Malaria."
    # class_names = ["have Malaria", "don't have Malaria"]
    st.success(string)
    # string = "Your report after analyzing your Cell image is you " + class_names[np.argmax(predictions)]
    

if st.button("Exit"):
        st.balloons()