import os
import pathlib
import pickle
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
        self.test_classes = test_classes
        self.test_images = test_images
        self.test_url_strings = test_url_strings

    def model_conversion(self): #MobileNet is not build through sequential API, so we need to convert it to sequential
        self.feature_model = tf.keras.applications.Xception()
        print("Model loaded")

    def TFconverter(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.feature_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        model_converter_file = pathlib.Path(model_converter)
        model_converter_file.write_bytes(tflite_model)

    def TFinterpreter(self):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_converter)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def Inference(self, img):
        input_shape = self.input_details[0]['shape']
        input_data = np.expand_dims(img, axis=0).astype(np.float32)

        assert np.array_equal(input_shape, input_data.shape), "Input tensor hasn't correct dimension"

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def extract_features(self):
        if not os.path.exists(n_neighbour_weights):
            self.test_features = np.array(
                            [self.Inference(img) for img in self.train_images]
                                        )
            self.test_features = self.test_features.reshape(self.test_features.shape[0],-1)
            self.neighbor = NearestNeighbors(
                                        n_neighbors = 20,
                                        )
            self.neighbor.fit(self.test_features)
            pickle.dump(self.neighbor, open(n_neighbour_weights, 'wb'))
        else:
            self.neighbor = pickle.load(open(n_neighbour_weights, 'rb'))

    def run(self):
        if not os.path.exists(model_converter):
            self.model_conversion()
            self.TFconverter()
        self.TFinterpreter()    
        self.extract_features()

    def predict_neighbour(self, dogimage, img_path):
        # update_db(img_path, lost_table)
        n_neighbours = {}
        data = self.Inference(dogimage)
        result = self.neighbor.kneighbors(data)[1].squeeze()
        result = nearest_neighbour_prediction(result, self.test_classes)
        fig=plt.figure(figsize=(8, 8))
        fig.add_subplot(2, 3, 1)
        plt.title('Input Image')
        plt.imshow(dogimage)
        for i in range(thres_neighbours):
            neighbour_img_id = result[i]
            fig.add_subplot(2, 3, i+2)
            plt.title('Neighbour {}'.format(i+1))
            plt.imshow(self.test_images[neighbour_img_id])
            label = self.test_classes[neighbour_img_id]
            print("Neighbour image {} label : {}".format(i+1, int(label)))

            n_neighbours["neighbour {}".format(i+1)] = "{}".format(self.test_url_strings[neighbour_img_id])
        plt.show()

        return n_neighbours

# model = DogSimDetector()
# model.run()