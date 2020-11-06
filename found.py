import os
import io
import json
import requests
import cv2 as cv
import pandas as pd
import base64
import numpy as np
from PIL import Image
from variables import *
from tensorflow.keras.preprocessing.image import img_to_array
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True

from util import *
from flask import Flask
from flask import jsonify
from flask import request

from sqlalchemy import create_engine
import sqlalchemy

'''
        python -W ignore found.py
        
        This flask API use to upload details when someone found a dog.
'''

app = Flask(__name__)

def preprocess_image(image, target_size): # Get the image, then resize and convert 3D tensor to 4D tensor
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

def get_image_path():
    img_arr = os.listdir(found_img_dir)
    if len(img_arr) > 0:
        img_idx = max([(int(os.path.split(img_path)[-1].split('.')[0])) for img_path in img_arr]) + 1
        img_name = str(img_idx)+'.png'
        img_path = os.path.join(found_img_dir,img_name)
    else:
        img_path = os.path.join(found_img_dir,'1.png') # return /home/isuru1997/Projects and Codes/SLIIT projects/Dog Breed Predictions/Found Images/1.png
    return img_path
    
def update_found_table(img_path): # update the image url in the data base
    engine = create_engine(db_url)
    if table_name in sqlalchemy.inspect(engine).get_table_names():
        data = pd.read_sql_table(table_name, db_url)
        df_length = len(data.values)
        data.loc[df_length+1] = img_path
        with engine.connect() as conn, conn.begin():
            data.to_sql(table_name, conn, if_exists='append', index=False)
    else:
        print("Create a Table named {}".format(table_name))


@app.route("/found", methods=["POST"])
def found():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    img_path = get_image_path

    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image)
    cv2.imwrite(img_path, processed_image) # save uploaded data into 
    update_found_table(img_path)

    response = {
        'uploaded_image': 'Successfully Uploaded'
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host=host, port= port, threaded=False)ss
