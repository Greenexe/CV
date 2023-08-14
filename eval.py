import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
###################### TENSORFLOW CONFIG ###########################
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
#############MLIR############################
#tf.config.experimental.enable_mlir_graph_optimization()
#tf.config.experimental.enable_mlir_bridge()
####################################################################
def set_seed(seed_value=123):
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    os.environ['HOROVOD_FUSION_THRESHOLD']='0'
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)
    from numpy.random import seed
    seed(seed_value)
    tf.random.set_seed(seed_value)
set_seed(666)

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from keras.utils.vis_utils import plot_model

from tensorflow import keras
from sklearn import preprocessing

from matplotlib import pyplot as plt

####local imports:
sys.path.append("/content/guided-diffusion-keras/guided_diffusion")
from denoiser import get_network
from utils import batch_generator, plot_images, get_data, preprocess
from diffuser import Diffuser

#########
# CONFIG:
#########


image_size = 32 #32
num_channels = 3
epochs = 300
class_guidance = 3
dataset_classes_count = 100

# architecture
channels = 64
channel_multiplier = [1, 2, 3]
block_depth = 3
emb_size = dataset_classes_count  # CLIP/label embedding 32
num_classes = 12  # placeholder  - 12
attention_levels = [0, 1, 0]

embedding_dims = dataset_classes_count #32
embedding_max_frequency = 1000.0

precomputed_embedding = True #False
save_in_drive = False
widths = [c * channels for c in channel_multiplier]

###train process config:
batch_size = 64
num_imgs = 100 #num imgs to test on - should be a square - 25, 64, 100 etc.
row = int(np.sqrt(num_imgs))

learning_rate = 0.0003

MODEL_NAME = "model_test_imgnet100"
file_name = "imageNet100.npz"

home_dir = MODEL_NAME

if not os.path.exists(home_dir):
    os.mkdir(home_dir)

model_path = os.path.join(home_dir, MODEL_NAME + ".h5")


##################################
###########Loading Data And Model:
##################################

if file_name == "cifar10":
    (train_data, train_label_embeddings), (_, _) = cifar10.load_data()
    #0 is used as a unconditional embedding:
    train_label_embeddings = train_label_embeddings + 1
elif file_name == "cifar100":
    (train_data, train_label_embeddings), (_, _) = cifar100.load_data()
    #0 is used as a unconditional embedding:
    train_label_embeddings = train_label_embeddings + 1
elif file_name == "fashion_mnist":
    (train_data, train_label_embeddings), (_, _) = fashion_mnist.load_data()
    train_data = train_data[:, :, :, None] #add extra dim at the end
    train_label_embeddings = train_label_embeddings[:, None]
    train_label_embeddings = train_label_embeddings + 1

else:
    #load the data from a npz file:
    train_data, train_label_embeddings = get_data(npz_file_path=file_name, prop=0.6, captions=False)

print(train_data.shape)

'''
if precomputed_embedding:
    labels = train_label_embeddings[:num_imgs]
else:
    labels = np.array([[i] * row for i in np.arange(row)]).flatten()[:, None]
'''
# Labels for all classes
labels = range(dataset_classes_count)[:num_imgs]
labels = np.array(labels, dtype=np.uint8)
labels = np.reshape(labels, (-1, 1))
enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
enc.fit(labels)
labels = enc.transform(labels).toarray()
labels = np.array(labels, dtype=np.uint8)
#

np.random.seed(100)
rand_image = np.random.normal(0, 1, (num_imgs, image_size, image_size, num_channels))


autoencoder = get_network(image_size,
                            widths,
                            block_depth,
                            num_classes=num_classes,
                            attention_levels=attention_levels,
                            emb_size=emb_size,
                            num_channels=num_channels,
                            precomputed_embedding=precomputed_embedding)

autoencoder.compile(optimizer="adam", loss="mae")
autoencoder.load_weights(model_path)

##################
#Some data checks:
##################

print("Number of pamaters is {0}".format(autoencoder.count_params()))


diffuser = Diffuser(autoencoder,
                    class_guidance=class_guidance,
                    diffusion_steps=35)

print(" Generating images:")
diffuser.denoiser = autoencoder
imgs = diffuser.reverse_diffusion(rand_image, labels)
plot_images(imgs, save_name="final_eval", nrows=int(np.sqrt(len(imgs))))