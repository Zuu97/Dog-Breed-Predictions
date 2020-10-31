import os
classes = [ '0', '1', '2', '3', '4']
batch_size = 12
valid_size = 4
color_mode = 'rgb'
width = 224
height = 224
target_size = (width, height)
input_shape = (width, height, 3)
shear_range = 0.2
zoom_range = 0.3
rotation_range = 30
shift_range = 0.2
rescale = 1./255
dense_1 = 512
dense_2 = 256
dense_3 = 64
num_classes = 5
epochs = 20
verbose = 1
val_split = 0.15

host = '0.0.0.0'
port = 5000
found_table = 'found_dog'
lost_table = 'lost_dog'
found_table = 'foundImages'
root_password = '1234'
db_url = 'mysql+pymysql://root:{}@localhost:3306/doggy_similarity'.format(root_password)
local_url = 'http://0.0.0.0:5000/predict'

# data directories and model paths
found_img_dir = os.path.join(os.getcwd(),'data/Found Dogs')
lost_img_dir = os.path.join(os.getcwd(),'data/Lost Dogs')
train_dir = os.path.join(os.getcwd(), 'data/Train images/')
test_dir = os.path.join(os.getcwd(), 'data/Test images/')
test_data_path = 'data/weights/Test_data.npz'
train_data_path = 'data/weights/Train_data.npz'
model_weights = "data/weights/model_weights.h5"
model_converter = "data/weights/model.tflite"
n_neighbour_weights = 'data/weights/nearest_neighbour.sav'