import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 1 - Only errors&warnings 2 - Only errors 3 - nothing

import tensorflow as tf
import numpy as np
import pandas as pd
###################### TENSORFLOW CONFIG ###########################
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
#############MLIR############################
#tf.config.experimental.enable_mlir_graph_optimization()
#tf.config.experimental.enable_mlir_bridge()
####################################################################
__mixed_prec = True
if __mixed_prec:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


def settings():
    epochs = 1000
    batch_size = 1024
    load_old = True

    dataX = "IoT_tabX_2.csv"
    dataY = "IoT_tabY_2.csv"

    model_name = "IoT_test1"

    return [epochs, batch_size, load_old], [dataX, dataY], [model_name]

def load_data(pathX:str, pathY:str, lstm:bool=False, val_split:float=None):
    # Load settings
    train_settings, *_ = settings()

    # Load dataset
    data_X = pd.read_csv(pathX,header=None)
    data_Y = pd.read_csv(pathY,header=None)

    # Convert to numpy
    data_X = np.array(data_X, 
                      dtype=np.float16 if __mixed_prec else np.float32
                      )
    data_Y = np.array(data_Y,
                      dtype=np.int32
                      )
    
    # Convert to Data Sequence
    if lstm:
        data_X = np.reshape(data_X, (-1, 1, data_X.shape[1]))
        data_Y = np.reshape(data_Y, (-1, data_Y.shape[1]))
    else:
        data_X = np.reshape(data_X, (-1, data_X.shape[1]))
        data_Y = np.reshape(data_Y, (-1, 1))

    def compute_data(data_X, data_Y):
        # Get metadata
        dataXshape = data_X.shape
        dataYshape = data_Y.shape
        dataXmin = data_X.min()
        dataXmax = data_X.max()

        # Print shapes
        print("Data X shape:", data_X.shape)
        print("Data Y shape:", data_Y.shape)

        # Convert to TF dataset
        ds_X = tf.data.Dataset.from_tensor_slices((data_X))
        ds_Y = tf.data.Dataset.from_tensor_slices((data_Y))
        train_ds = tf.data.Dataset.zip((ds_X, ds_Y))

        # Batch the data
        train_ds = train_ds.batch(batch_size=train_settings[1])

        # Prepare the dataset
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        return train_ds, [dataXshape, dataYshape, dataXmin, dataXmax]

    if val_split == None:
        # Shuffle both arrays with the same random state
        random_state = np.random.RandomState(42)
        random_state.shuffle(data_X)
        random_state.shuffle(data_Y)

        return compute_data(data_X=data_X, data_Y=data_Y)
    else:
        split_number = int(data_X.shape[0] * val_split)

        # Shuffle both arrays with the same random state
        random_state = np.random.RandomState(42)
        random_state.shuffle(data_X)
        random_state.shuffle(data_Y)

        data_X_val = data_X[split_number:]
        data_Y_val = data_Y[split_number:]
        data_X_tr = data_X[:split_number]
        data_Y_tr = data_Y[:split_number]
        train = compute_data(data_X_tr, data_Y_tr)
        val = compute_data(data_X_val, data_Y_val)
        return [train[0], val[0]], [train[1], val[1]]

def normalize(train_ds, metadata:tuple):
    return train_ds

def create_model(metadata:tuple):
    # Helpers
    Dense = tf.keras.layers.Dense
    Bidirectional = tf.keras.layers.Bidirectional
    LSTM = tf.keras.layers.LSTM
    Input = tf.keras.layers.Input

    # Model
    model = tf.keras.Sequential()
    model.add(Input(shape=metadata[0][1:]))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    # Flatten
    model.add(Bidirectional(LSTM(64, return_sequences=False)))

    # Flat
    model.add(Dense(64,activation='tanh'))
    model.add(Dense(24,activation='tanh'))

    # Output layer
    model.add(Dense(1,activation='sigmoid'))


    # Compile model
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    opt = tf.keras.optimizers.Nadam(learning_rate=0.001)

    model.compile(
        loss=loss, 
        optimizer=opt, 
        metrics=['binary_accuracy']
        )

    return model

def create_callbacks(name:str="model", mode:str="train") -> list:
    callbacks = []

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            "./weights/"+name+"_acc.keras",
            monitor="binary_accuracy" if mode != "val" else "val_binary_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
        )
        )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            "./weights/"+name+"_loss.keras",
            monitor="loss" if mode != "val" else "val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
        )
        )

    return callbacks

def main():
    # Load settings
    train_settings,\
    data_settings,\
    train_metadata = settings()

    # Create dataset
    print("Loading data...")
    dataset, metadata = load_data(data_settings[0], data_settings[1], lstm=True, val_split=0.7)
    dataset = normalize(dataset, metadata[0])

    # Create model
    model = create_model(metadata[0])

    # Create callbacks
    callbacks = create_callbacks(train_metadata[0], mode="val")

    # Load old
    if train_settings[2]:
        print("Loading model...")
        try:
            model.load_weights("./weights/"+train_metadata[0]+"_loss.keras")
            print(">Loaded old weights.")
        except:
            print(">Cannot load old weights.")

    # Train the model
    model.fit(dataset[0], validation_data=dataset[1],
              epochs=train_settings[0],
              batch_size=train_settings[1],
              callbacks=callbacks,
              verbose=1
              )
    
    # Evaluate the model
    model.load_weights("./weights/"+train_metadata[0]+"_loss.keras")
    model.evaluate(dataset[1], batch_size=train_settings[1])

def prepare():
    # Create needed folders
    try:
        os.makedirs("./weights/")
    except:
        pass
    try:
        os.makedirs("./plots/")
    except:
        pass

def set_seed(seed_value:int=123):
    # set environment variables
    os.environ['PYTHONHASHSEED']=str(seed_value)
    os.environ['HOROVOD_FUSION_THRESHOLD']='0'
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # 2. Set pseudo-random generators at a fixed value
    import random
    random.seed(seed_value)
    from numpy.random import seed
    seed(seed_value)
    tf.random.set_seed(seed_value)

if __name__ == "__main__":
    set_seed(1)
    prepare()
    main()