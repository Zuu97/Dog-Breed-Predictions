import os
import json
import pandas as pd
import numpy as np
import cv2 as cv
from variables import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import DogSimDetector
import logging
logging.getLogger('tensorflow').disabled = True
from tensorflow.keras.preprocessing.image import img_to_array

import requests
from PIL import Image
from util import *
from flask import Flask
from flask import jsonify
from flask import request
import werkzeug
'''
        python -W ignore app.py
'''

app = Flask(__name__)

model = DogSimDetector()
model.run()

if not os.path.exists(found_img_dir):
    os.makedirs(found_img_dir)

if not os.path.exists(lost_img_dir):
    os.makedirs(lost_img_dir)

def get_image_path():
    img_arr = os.listdir(lost_img_dir)
    if len(img_arr) > 0:
        img_idx = max([(int(os.path.split(img_path)[-1].split('.')[0])) for img_path in img_arr]) + 1
        img_name = str(img_idx)+'.png'
        img_path = os.path.join(lost_img_dir,img_name)
    else:
        img_path = os.path.join(lost_img_dir,'1.png') 
    return img_path


def preprocess_image(image, target_size):
    if image.shape[-1] == 1:
        return False
    else:
        image = cv.resize(image, target_size)
        return image

def save_image(image, img_path):
        img = Image.fromarray(image)
        img.save(img_path)

@app.route("/predict", methods=["POST"])
def predict():

      dogimagefile= request.files['image'].read()
      dogimage = np.fromstring(dogimagefile, np.uint8)
      dogimage = cv.imdecode(dogimage,cv.IMREAD_COLOR) 
      dogimage = preprocess_image(dogimage, target_size)
      if dogimage.any():
        img_path = get_image_path()
        save_image(dogimage, img_path)   
        dogimage = dogimage*rescale
        n_neighbours = model.predict_neighbour(dogimage, img_path)

        # response = {
        #             "n_neighbours": n_neighbours
        #            }
        response = requests.post(local_url, files=n_neighbours)
        return jsonify(response)

      else:
        return "Please Insert RGB image of your DOG !"

if __name__ == "__main__": 
    app.run(debug=True, host=host, port= port, threaded=False, use_reloader=False)