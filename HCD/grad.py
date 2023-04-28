import tensorflow as tf
import numpy as np
from PIL import Image
# from weights import download_weights
# from model import download_model
import cv2
import gradio as gr

# def getCropImgs(img, needRotations=False):
#     # img = img.convert('L')
#     z = np.asarray(img, dtype=np.int8)
#     c = []
#     for i in range(3):
#         for j in range(4):
#             crop = z[512 * i:512 * (i + 1), 512 * j:512 * (j + 1), :]

#             c.append(crop)
#             if needRotations:
#                 c.append(np.rot90(np.rot90(crop)))

#     # os.system('cls')
#     # print("Crop imgs", c[2].shape)

    # return c

def load_model():
    model = tf.keras.models.load_model("./my_model")
    return model

model = load_model()

from PIL import Image, ImageOps
def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)

    return prediction

def main():
  image=gr.inputs.Image(shape=(64,64))
  model = load_model()
  pred = import_and_predict(image, model)
  class_name=['Benign with Density=1','Malignant with Density=1','Benign with Density=2','Malignant with Density=2','Benign with Density=3','Malignant with Density=3','Benign with Density=4','Malignant with Density=4']
  
#   def predict_img(img):
#     img=preprocess(img)
#     img=img/255.0
#     im=img.reshape(-1,224,224,3)
#     # im = im.reshape(1, 5024, 5024)
#     pred=model.predict(im)[0]
#     return {class_name[i]:float(pred[i]) for i in range(8)}
  
  gr.Interface(fn=image,inputs=image,outputs=pred,capture_session=True).launch(debug='True',share=True)
  
if __name__=='__main__':
    main()
    
