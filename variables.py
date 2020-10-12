import os
classes = [ '0', '1', '2', '3', '4']
# , '5', '6', '7', '8', '9'
batch_size = 8
valid_size = 4
color_mode = 'rgb'
width = 224
height = 224
target_size = (width, height)
input_shape = (width, height, 3)
shear_range = 0.2
zoom_range = 0.15
rotation_range = 20
shift_range = 0.2
rescale = 1./255
dense_1 = 512
dense_2 = 256
dense_3 = 64
num_classes = 5
epochs = 10
verbose = 1
val_split = 0.15

host = '0.0.0.0'
port = 5000
table_name = 'doggy'
root_password = 'Isuru767922513'
db_url = 'mysql+pymysql://root:{}@localhost:3306/doggy_similarity'.format(root_password)

# data directories and model paths
train_dir = os.path.join(os.getcwd(), 'Train images/')
test_dir = os.path.join(os.getcwd(), 'Test images/')
test_data_path = 'weights/Test_data.npz'
train_data_path = 'weights/Train_data.npz'
model_weights = "weights/doggy_mobilenet.h5"
model_architecture = "weights/doggy_mobilenet.json"