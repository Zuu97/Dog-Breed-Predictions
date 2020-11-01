import tensorflow as tf
import numpy as np
import os
import numpy as np
import pandas as pd
import cv2 as cv
from sklearn.utils import shuffle
from sqlalchemy import create_engine
import sqlalchemy

from variables import*

def preprocessing_function(img):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def image_data_generator():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    rotation_range = rotation_range,
                                    shear_range = shear_range,
                                    zoom_range = zoom_range,
                                    width_shift_range=shift_range,
                                    height_shift_range=shift_range,
                                    horizontal_flip = True,
                                    validation_split= val_split,
                                    preprocessing_function=preprocessing_function
                                    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    preprocessing_function=preprocessing_function
                                    )


    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = classes,
                                    subset = 'training',
                                    shuffle = True)

    validation_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = valid_size,
                                    classes = classes,
                                    subset = 'validation',
                                    shuffle = True)

    test_generator = test_datagen.flow_from_directory(
                                    test_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = classes,
                                    shuffle = False)

    return train_generator, validation_generator, test_generator
    
def load_test_data(data_path, save_path):
    data_name = os.path.split(save_path)[-1].split('_')[0]
    if not os.path.exists(save_path):
        print("{} Images Saving".format(data_name))
        images = []
        classes = []
        url_strings = []
        dog_folders = os.listdir(data_path)
        for label in list(dog_folders):
            label_dir = os.path.join(data_path, label)
            label_images = []
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = preprocessing_function(img)
                img = cv.resize(img, target_size, cv.INTER_AREA).astype(np.float32)

                images.append(img)
                classes.append(int(label))
                url_strings.append(img_path)

        images = np.array(images).astype('float32')
        classes = np.array(classes).astype('float32')
        url_strings = np.array(url_strings)
        np.savez(save_path, name1=images, name2=classes, name3=url_strings)
    else:
        data = np.load(save_path, allow_pickle=True)
        images = data['name1']
        classes = data['name2']
        url_strings = data['name3']
        print("{} Images Loaded".format(data_name))

    classes, images, url_strings = shuffle(classes, images, url_strings)
    return classes, images, url_strings

def update_db(img_path, table_name):
    engine = create_engine(db_url)
    if table_name in sqlalchemy.inspect(engine).get_table_names():
        data = pd.read_sql_table(table_name, db_url)
        df_length = len(data.values)
        df.loc[df_length]['image path'] = img_path
        with engine.connect() as conn, conn.begin():
            data.to_sql(table_name, conn, if_exists='append', index=False)
    else:
        print("Create a Table named {}".format(table_name))

def nearest_neighbour_prediction(result, test_classes):
    labels = [int(test_classes[neighbour_img_id]) for neighbour_img_id in result[1:]]
    label = np.bincount(labels).argmax()
    print(np.bincount(labels))
    labels = np.array(labels)
    correct_idx = np.where(labels == label)[0]
    return result[correct_idx[:thres_neighbours+1]]