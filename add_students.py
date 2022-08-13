
import numpy as np
import json
import create_model
from data_preprocessing import data_prepare

def read_data(X_path="Final_Model/X_train.csv", y_path="Final_Model/y_train.csv", en_path="Final_Model/encoder.json"):
    X_train = np.genfromtxt(X_path, delimiter=',')
    y_train = np.genfromtxt(y_path, delimiter=',')
    with open(en_path, "r") as f:
        encoder = json.load(f)
    return X_train.reshape((-1, 224, 224, 3)), y_train, encoder


def training(model, X_train, y_train):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from tensorflow.keras.callbacks import TensorBoard
    import os 
    import utils
    base = r"Final_Model/"
    os.makedirs(base, exist_ok=True)
    checkpoint = ModelCheckpoint(os.path.join(base,"model.h5"), monitor='accuracy', verbose=1,
                                 save_best_only=True, mode='auto', period=1, save_freq='epoch')


    model.compile(optimizer = keras.optimizers.Adam(lr = 0.001, 
                                                    beta_1 = 0.9, 
                                                    beta_2 = 0.999), 
                              loss = 'sparse_categorical_crossentropy',  
                              metrics = ['accuracy'])


    reduce_lr = ReduceLROnPlateau(monitor = 'accuracy', patience = 5, verbose = 1, factor = 0.2, min_lr = 0.000001,
                                                mode = 'auto', cooldown = 0)

    #X_train, y_train, encoder = read_data()
    X_train_prepared = utils.preprocess_input(X_train.astype('float64'), version=2)
    train_ds = utils.data_augmentation(X_train_prepared, y_train)

    class StopTrainingCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            #print(logs)
            #logs = {'accuracy':0.0} if logs is None else logs
            ACCURACY_THRESHOLD = 1.0
            if(logs.get('accuracy') >= ACCURACY_THRESHOLD):
                self.model.stop_training = True

    stp = StopTrainingCallback()
    history = model.fit(train_ds, epochs=50,
                    callbacks=[checkpoint, reduce_lr, stp], verbose=0)
    
    return model

def add(first_time=False):
    if first_time:
        X_train, y_train, label_encoder, encoder = data_prepare()
        model = create_model.create_model(len(list(encoder.keys())))
        custom_model = training(model, X_train, y_train)
        return custom_model
    else:
        from tensorflow import keras
        model = keras.models.load_model("Final_Model/model.h5")
        X_train, y_train, encoder = read_data()
        label_encoder = {v:k for k,v in encoder.items()}
        X_train, y_train, label_encoder, encoder = data_prepare(old_encoder=label_encoder, 
                                                            X_train_old=X_train, y_train_old=y_train)
        model_ = create_model.create_model(len(list(encoder.keys())), model)
        custom_model = training(model_, X_train, y_train)
        return custom_model
