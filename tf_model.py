import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input, Flatten
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
        
        if not os.path.exists(model_weights):
            train_generator, validation_generator, test_generator = image_data_generator()
            self.test_generator = test_generator
            self.train_generator = train_generator
            self.validation_generator = validation_generator
            self.train_step = self.train_generator.samples // batch_size
            self.validation_step = self.validation_generator.samples // valid_size
            self.test_step = self.test_generator.samples // batch_size

        self.test_classes = test_classes
        self.test_images = test_images
        self.test_url_strings = test_url_strings

    def model_conversion(self):
        functional_model = tf.keras.applications.MobileNetV2(
                                                    # include_top=False,
                                                    weights="imagenet",
                                                    # input_shape=input_shape
                                                             )
        functional_model.trainable = False
        inputs = functional_model.input

        x = functional_model.layers[-2].output
        x = Dense(dense_1, activation='relu')(x)
        x = Dense(dense_1, activation='relu')(x)
        x = Dense(dense_2, activation='relu')(x)
        x = Dense(dense_3, activation='relu')(x)
        x = Dense(dense_3, activation='relu')(x)
        x = Dense(dense_3, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(
                inputs =inputs,
                outputs=outputs
                    )
        self.model = model
        self.model.summary()

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
        self.feature_model.save(model_weights)
        print("Feature CNN Model Saved")

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        self.feature_model = load_model(model_weights)
        print("Feature CNN Model Loaded")

    def Evaluation(self):
        Predictions = self.model.predict_generator(self.test_generator,steps=self.test_step)
        P = np.argmax(Predictions,axis=1)
        loss , accuracy = self.model.evaluate_generator(self.test_generator, steps=self.test_step)
        print("test loss : ",loss)
        print("test accuracy : ",accuracy)

    def run_MobileNet(self):
        self.model_conversion()
        self.train()
        self.Evaluation()

    def feature_extraction_model(self):
        inputs = self.model.input
        outputs = self.model.layers[-1].output
        feature_model = Model(
                        inputs =inputs,
                        outputs=outputs
                            )
        self.feature_model = feature_model
        # self.feature_model.summary()

    def TFconverter(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.feature_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(model_converter, 'wb') as file:
            file.write(tflite_model)

    def TFinterpreter(self):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_converter)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.neighbor = pickle.load(open(n_neighbour_weights, 'rb'))

    def Inference(self, img):
        input_shape = self.input_details[0]['shape']
        input_data = np.array([img], dtype=np.float32)

        assert np.array_equal(input_shape, input_data.shape), "Input tensor hasn't correct dimension"

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def extract_features(self):
        if not os.path.exists(n_neighbour_weights):
            self.test_features = self.feature_model.predict(self.test_images)
            self.neighbor = NearestNeighbors(n_neighbors = 6)
            self.neighbor.fit(self.test_features)
            pickle.dump(self.neighbor, open(n_neighbour_weights, 'wb'))

    def run(self):
        if not os.path.exists(model_converter):
            if not os.path.exists(model_weights):
                self.run_MobileNet()
                self.feature_extraction_model()
                self.save_model()
            else:
                self.load_model()
            self.extract_features()
            self.TFconverter()
        else:
            self.TFinterpreter()        
            print(self.test_classes)
    def predict_neighbour(self, dogimage, img_path):
        # update_db(img_path, lost_table)
        n_neighbours = {}
        data = self.Inference(dogimage)
        result = self.neighbor.kneighbors(data)[1].squeeze()
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

            n_neighbours["neighbour {}".format(i-1)] = "{}".format(self.test_url_strings[neighbour_img_id])
        plt.show()

        return n_neighbours

model = DogSimDetector()
model.run()