import tensorflow as tf
import numpy as np
import os
import numpy as np
import pandas as pd
import cv2 as cv
import base64
from sklearn.utils import shuffle
from sqlalchemy import create_engine
import sqlalchemy

from variables import*

def image_data_generator():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    rescale = rescale,
                                    rotation_range = rotation_range,
                                    shear_range = shear_range,
                                    zoom_range = zoom_range,
                                    width_shift_range=shift_range,
                                    height_shift_range=shift_range,
                                    horizontal_flip = True,
                                    validation_split= val_split
                                    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = rescale)


    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = classes,
                                    shuffle = True)

    validation_generator = test_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = valid_size,
                                    classes = classes,
                                    shuffle = True)

    test_generator = test_datagen.flow_from_directory(
                                    test_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = classes,
                                    shuffle = False)

    return train_generator, validation_generator, test_generator
def decode_byte_string(byte_string):
    byte_code = byte_string.encode("ascii")
    string_bytes = base64.b64decode(byte_code)
    return string_bytes.decode("ascii")

def encode_byte_string(url_string):
    string_bytes = url_string.encode("ascii")
    byte_code = base64.b64encode(string_bytes)
    return byte_code.decode("ascii")

def load_test_data(data_path, save_path):
    data_name = os.path.split(save_path)[-1].split('_')[0]
    if not os.path.exists(save_path):
        print("{} Images Saving".format(data_name))
        images = []
        classes = []
        byte_strings = []
        dog_folders = os.listdir(data_path)
        for label in list(dog_folders):
            label_dir = os.path.join(data_path, label)
            label_images = []
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)*rescale
                img = cv.resize(img, target_size)

                images.append(img)
                classes.append(int(label))
                byte_url = encode_byte_string(img_path)
                byte_strings.append(byte_url)

        images = np.array(images).astype('float32')
        classes = np.array(classes).astype('float32')
        byte_strings = np.array(byte_strings)
        np.savez(save_path, name1=images, name2=classes, name3=byte_strings)
    else:
        print("{} Images Loaded".format(data_name))
        data = np.load(save_path, allow_pickle=True)
        images = data['name1']
        classes = data['name2']
        byte_strings = data['name3']

    classes, images, byte_strings = shuffle(classes, images, byte_strings)
    return classes, images, byte_strings

def update_db(byte_string):
    engine = create_engine(db_url)
    if table_name in sqlalchemy.inspect(engine).get_table_names():
        data = pd.read_sql_table(table_name, db_url)
        df_length = len(data.values)
        data.loc[df_length+1] = decode_byte_string(byte_string)
        with engine.connect() as conn, conn.begin():
            data.to_sql(table_name, conn, if_exists='append', index=False)
    else:
        print("Create a Table named {}".format(table_name))

# test_classes, test_images, test_byte_strings = load_test_data(test_dir, test_data_path)
# print(test_byte_strings.shape)
# print(test_byte_strings.tolist().index('VGVzdCBpbWFnZXMvNFxuMDIwODYwNzlfMTc1OS5qcGc='))