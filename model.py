import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import base64
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
        test_classes, test_images, test_byte_strings = load_test_data(test_dir, test_data_path)
        train_classes, train_images, _ = load_test_data(train_dir, train_data_path)
        print(test_byte_strings) 
        
        if not (os.path.exists(model_architecture)  and os.path.exists(model_weights)):
            train_generator, validation_generator, test_generator = image_data_generator()
            self.test_generator = test_generator
            self.train_generator = train_generator
            self.validation_generator = validation_generator
            self.train_step = self.train_generator.samples // batch_size
            self.validation_step = self.validation_generator.samples // valid_size
            self.test_step = self.test_generator.samples // batch_size

        self.train_classes = train_classes
        self.train_images = train_images
        self.test_classes = test_classes
        self.test_images = test_images
        self.test_byte_strings = test_byte_strings

    def model_conversion(self): #MobileNet is not build through sequential API, so we need to convert it to sequential
        mobilenet_functional = tf.keras.applications.MobileNet()
        # mobilenet_functional = tf.keras.applications.MobileNetV2()
        model = Sequential()
        for layer in mobilenet_functional.layers[:-1]:# remove the softmax in original model. because we have only 3 classes
            layer.trainable = False
            model.add(layer)
        model.add(Dense(dense_1, activation='relu'))
        model.add(Dense(dense_2, activation='relu'))
        model.add(Dense(dense_2, activation='relu'))
        model.add(Dense(dense_3, activation='relu'))
        model.add(Dense(dense_3, activation='relu'))
        model.add(Dense(dense_3, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # model.summary()
        self.model = model

    def train(self):
        self.model.compile(
                          optimizer='Adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy']
                          )
        self.model.fit_generator(
                          self.train_generator,
                          steps_per_epoch= self.train_step,
                          validation_data= self.validation_generator,
                          validation_steps = self.validation_step,
                          epochs=epochs,
                          verbose=verbose
                        )

    def save_model(self):
        print("Model Saving !")
        model_json = self.model.to_json()
        with open(model_architecture, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights)

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        json_file = open(model_architecture, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(model_weights)

        # self.model.compile(
        #                    loss='categorical_crossentropy', 
        #                    optimizer='Adam', 
        #                    metrics=['accuracy']
        #                    )
        print("Model Loaded")

    def predict_MobileNet(self):
        Predictions = self.model.predict_generator(self.test_generator,steps=self.test_step)
        P = np.argmax(Predictions,axis=1)
        loss , accuracy = self.model.evaluate_generator(self.test_generator, steps=self.test_step)
        print("test loss : ",loss)
        print("test accuracy : ",accuracy)
        # print(Predictions.shape)
        # print("Predictions : ",Predictions)

    def run_MobileNet(self):
        if os.path.exists(model_weights):
            self.load_model()
        else:
            self.model_conversion()
            self.train()
            self.save_model()
        # self.predict_MobileNet()

    def feature_extractor(self):
        feature_model = Sequential()
        for layer in self.model.layers[:-4]:# remove last 4 layers in original model. because we have only 3 classes
            layer.trainable = False
            feature_model.add(layer)
        self.feature_model = feature_model

    def extract_features(self):
        self.test_features = self.feature_model.predict(self.test_images)
        self.train_features = self.feature_model.predict(self.train_images)
        print("Done")

    def predict_neighbour(self, byte_url):
        # update_db(byte_url)
        if byte_url in self.test_byte_strings:
            n_neighbours = {}
            img_id = self.test_byte_strings.tolist().index(byte_url)
            data = self.test_features[img_id]
            neighbor = NearestNeighbors(n_neighbors = 6)
            neighbor.fit(self.test_features)
            result = neighbor.kneighbors([data])[1].squeeze()
            fig=plt.figure(figsize=(8, 8))
            fig.add_subplot(2, 3, 1)
            plt.title('Input Image')
            plt.imshow(self.test_images[img_id])
            print("\nInput image label : {}".format(self.test_classes[img_id]))
            for i in range(2, 7):
                neighbour_img_id = result[i-1]
                fig.add_subplot(2, 3, i)
                plt.title('Neighbour {}'.format(i-1))
                plt.imshow(self.test_images[neighbour_img_id])
                print("Neighbour image {} label : {}".format(i-1, self.test_classes[neighbour_img_id]))

                base64_string = base64.b64encode(self.test_images[neighbour_img_id])
                n_neighbours['neighbour ' + str(i-1)] = base64_string

            plt.show()

            return n_neighbours

        else:
            print("Byte Url doesn't exists")

    def run_feature_model(self):
        self.feature_extractor()
        self.extract_features()

    def run(self):
        self.run_MobileNet()
        self.run_feature_model()

# if __name__ == "__main__":
#     model = DogSimDetector()
#     model.run()
#     model.predict_neighbour(model.test_byte_strings[10])
