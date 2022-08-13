
from PIL import Image
import tensorflow as tf
from mtcnn import MTCNN
import numpy as np


# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
detector.min_face_size = 70


def bbox_scaling(x1, y1, x2, y2, width, height, factor=0.4):
        x1_new = x1 - (factor/2)*width
        y1_new = y1 - (factor/2)*height
        x2_new = x2 + (factor/2)*width
        y2_new = y2 + (factor/2)*height
        if x1_new < 0:
            x1_new = 0
        if y1_new < 0:
            y1_new = 0
        return int(round(x1_new)), int(round(y1_new)), int(round(x2_new)), int(round(y2_new))

    
    
def extract_faces(image, required_size=(224, 224), just_one_face=True):
    
    
    results = detector.detect_faces(image)
    if results:
        if not just_one_face:
            faces = []
            axis = []
            for result in results:
                # extract the bounding box from the first face
                x1, y1, width, height = result['box']
                # bug fix
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                x1, y1, x2, y2 = bbox_scaling(x1, y1, x2, y2, width, height, factor=0.3)
                face = image[y1:y2, x1:x2]
                #resize pixels to the model size
                im = Image.fromarray(face)
                im = im.resize(required_size)
                face = np.asarray(im)
                faces.append(face)
                axis.append([x1, y1, x2, y2])
        else:
            x1, y1, width, height = results[0]['box']
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            x1, y1, x2, y2 = bbox_scaling(x1, y1, x2, y2, width, height, factor=0.3)
            face = image[y1:y2, x1:x2]
            im = Image.fromarray(face)
            im = im.resize(required_size)
            face = np.asarray(im)
            axis = [x1, y1, x2, y2]
    else:
        return (None, None)

    if just_one_face:
        return face, axis
    else:
        return np.asarray(faces), np.asarray(axis)
