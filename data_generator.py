import tensorflow as tf
import numpy as np
import os

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        mode,
        batch_size,
        dataset_path,
        latent_size,
        global_image_size,
        shuffle=True,
    ):
        self.mode = mode
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dataset_path = dataset_path

        self.dataset = []
        for folders, sub_folders, files in os.walk(dataset_path):
            # For each name in the files
            for filename in files:
                if filename[-4:] == "JPEG":
                    print("dasd")
                    self.dataset.append(os.path.join(folders, filename))

        self.latent_size = latent_size
        self.global_image_size = global_image_size
        self.on_epoch_end()

    def _load_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, (self.global_image_size,self.global_image_size))
        #image = data_augmentation(image) if self.mode == "train" else image
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5
        return image

    def _load_latent(self, path):
        latent = np.loadtxt(path, delimiter=",")
        latent = np.reshape(latent, (-1))
        latent = tf.cast(latent, tf.float32)
        latent = latent / np.max(np.abs(latent))
        return latent

    def on_epoch_end(self):
        self.index = tf.range(len(self.dataset))
        if self.shuffle == True:
            tf.random.shuffle(self.index)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.index[idx * self.batch_size : (idx + 1) * self.batch_size]
        datset_keys = [self.dataset[k] for k in indexes]
        (x_train, y_train) = self.__data_generation(datset_keys)
        return x_train, y_train

    def __data_generation(self, index):
        batch_images = []
        batch_latent = []
        for idx, i in enumerate(index):
            image = self._load_image(i)
            latent = self._load_latent(i[:-3]+"csv")
            batch_images.append(image)
            batch_latent.append(latent)
        batch_images = np.array(batch_images)
        batch_latent = np.array(batch_latent)

        #randomness
        #random_noise = np.random.normal(0, 1, batch_latent.shape)
        #batch_latent += random_noise * 0.33
        ###########
        
        return batch_latent, batch_images
