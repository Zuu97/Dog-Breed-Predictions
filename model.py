import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from util import *
from variables import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DogSimDetector(object):
    def __init__(self):
        test_classes, test_images, test_url_strings = load_test_data(test_dir, test_data_path)
        train_classes, train_images, _ = load_test_data(train_dir, train_data_path)
        
        # if not (os.path.exists(model_architecture)  and os.path.exists(model_weights)):
        #     train_generator, validation_generator, test_generator = image_data_generator()
        #     self.test_generator = test_generator
        #     self.train_generator = train_generator
        #     self.validation_generator = validation_generator
        #     self.train_step = self.train_generator.samples // batch_size
        #     self.validation_step = self.validation_generator.samples // valid_size
        #     self.test_step = self.test_generator.samples // batch_size

        self.train_classes = train_classes
        self.train_images = train_images
        self.test_classes = test_classes
        self.test_images = test_images
        self.test_url_strings = test_url_strings

    def model_conversion(self): #MobileNet is not build through sequential API, so we need to convert it to sequential
        self.model = tf.keras.applications.MobileNetV2()
        print("Model loaded")

    def extract_features(self):
        self.test_features = self.model.predict(self.test_images)
        self.neighbor = NearestNeighbors(n_neighbors = 6)
        self.neighbor.fit(self.test_features)

    def predict_neighbour(self, dogimage, img_path):
        # update_db(img_path, lost_table)
        n_neighbours = {}
        data = self.model.predict(np.array([dogimage])).squeeze()
        result = self.neighbor.kneighbors([data])[1].squeeze()
        fig=plt.figure(figsize=(8, 8))
        fig.add_subplot(2, 3, 1)
        plt.title('Input Image')
        plt.imshow(dogimage)
        for i in range(2, 7):
            neighbour_img_id = result[i-1]
            fig.add_subplot(2, 3, i)
            plt.title('Neighbour {}'.format(i-1))
            plt.imshow(self.test_images[neighbour_img_id])
            label = self.test_classes[neighbour_img_id]
            print("Neighbour image {} label : {}".format(i-1, int(label)))

            n_neighbours['neighbour ' + str(i-1)] = "{}".format(self.test_url_strings[neighbour_img_id])
        plt.show()

        return n_neighbours

    def run(self):
        self.model_conversion()
        self.extract_features()
