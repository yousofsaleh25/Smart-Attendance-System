

from PIL import Image
from PIL import ImageEnhance
from face_detection import extract_faces
import numpy as np
import os


def data_save(X_train, y_train, encoder):
    import json
    np.savetxt("Final_Model/X_train.csv", X_train.ravel(), delimiter=",")
    np.savetxt("Final_Model/y_train.csv", y_train.ravel(), delimiter=",")
    json = json.dumps(encoder)
    with open("Final_Model/encoder.json", "w") as f:
        f.write(json)

def reverse_encoder(encoder):
    encoder = {v:k for k,v in sorted(encoder.items(), key=lambda item:item[1])}
    return encoder

def data_prepare(path='images/', old_encoder={}, X_train_old=[], y_train_old=[], save=True):
    last_names = list(old_encoder.keys())
    #print(X_train_old)
    #print(len(X_train_old))
    X_train = list()
    y_train = list()
    label_encoder = dict()
    if len(X_train_old) > 0:
        print(last_names)
        print(len(last_names))
        count_id = len(last_names)
    else:
        count_id = 0
    for root, dirs, files in os.walk("images/", topdown=False):

        for name in files:
            #print(root)
            image_path = os.path.join(root, name)
            image_label = os.path.basename(root)
            #print(image_label)
            #print(os.path.join(root, name))
            if image_label in last_names:
                continue
            if image_label not in label_encoder.keys():
                label_encoder[image_label] = count_id
                count_id += 1
            
            im = Image.open(image_path)
            enh = ImageEnhance.Contrast(im)
            enh = enh.enhance(1.15)
            if enh.mode != 'RGB':
                enh.convert('RGB')
            face, _ = extract_faces(np.asarray(enh), required_size=(224, 224), just_one_face=True)
            if face is not None:
                X_train.append(face)
                y_train.append(label_encoder[image_label])
            #print(im.format, im.size, im.mode)
            #print(type(im))
    #     for directory in dirs:
    #         print(directory)
    label_encoder.update(old_encoder)
    encoder = {v:k for k,v in label_encoder.items()}
    X_train.extend(X_train_old)
    y_train.extend(y_train_old)
    #print(np.asarray(X_train).shape)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    if save:
        data_save(X_train, y_train, encoder)
    return X_train, y_train, label_encoder, encoder
