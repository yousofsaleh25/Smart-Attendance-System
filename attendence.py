
import json
import pandas as pd
from tensorflow import keras
import numpy as np
import os
from tqdm import tqdm
import cv2
model = keras.models.load_model("Final_Model/model.h5")
from face_detection import extract_faces
from utils import preprocess_input
import arabic_reshaper
from PIL import Image, ImageFont, ImageDraw
from bidi.algorithm import get_display
import arabic_reshaper
from PIL import Image, ImageFont, ImageDraw
from bidi.algorithm import get_display


def write_txt(frame, txt, pos):
    fontpath = r"C:\Users\Dell\Desktop\py\Face_Detection\Smart Attendence System\Final_Model\Language_Arabic_Text\arial.ttf"
    font = ImageFont.truetype(fontpath, 18)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    reshaped_text = arabic_reshaper.reshape(txt)
    bidi_text = get_display(reshaped_text)
    draw.text(pos, bidi_text, font=font)
    return np.asarray(img_pil)

#font = ImageFont.truetype(fontpath, 20)
def take_attendence(video_path="test_video/test.mp4", show_video=False):
    cap = cv2.VideoCapture(video_path)
    # Setup Video writer
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #videowriter = cv2.VideoWriter(os.path.join("predicted_video", "result.avi"),
     #             cv2.VideoWriter_fourcc('P', 'I', 'M', '1'), fps, (width, height),
     #                             isColor=True)



    with open("Final_Model/encoder.json", 'r') as f:
        encoder = json.load(f)
        #print(encoder)

    attendence = {k:10 for k,v in encoder.items()}
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_index in tqdm(range(num_frames)):
        ret, frame = cap.read()
        #print(ret)
        frame = frame[:, :, ::-1]
        #faces = extract_faces_axis(frame)
        faces_img, faces = extract_faces(frame, required_size=(224, 224),
                                  just_one_face=False)
        #print(np.any(faces_img == None))
        if faces_img is not None:
            faces_img = preprocess_input(np.array(faces_img, dtype="float64"), version=2)
            labels_ = model.predict(faces_img)
            #print(labels)
            labels = np.argmax(labels_, axis=1)
            th = labels_[0][labels[0]]
            #print(th)
            if th > 0.60:
                for face, label in  zip(faces, [labels[0]]):
                    #print(face_img.shape)
                    #label = custom_vgg_model.predict([face_img])
                    name = encoder[str(label)]
                    attendence[str(label)] -= 1
                    #print(attendence[str(label)])
                    if show_video:
                        x1, y1, x2, y2 = face
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        try:
                            cv2.rectangle(frame[:, :, ::-1], (x1, y1), (x2, y2), (255, 255, 255), 1)
                        except:
                            pass
                        frame = write_txt(frame, name, (x1, y1-15))
                        #font = cv2.FONT_HERSHEY_SIMPLEX
                        #cv2.putText(frame[:, :, ::-1], name, (x1, y1-10), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
                    #print(name)
        if show_video:
            cv2.imshow('Video Player' , frame[:, :,::-1])
        #videowriter.write(frame[:, :,::-1])
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    attendence = {encoder[str(k)]:(True if v <= 0 else False)\
                   for k,v in attendence.items()}
    cap.release()
    cv2.destroyAllWindows()
    #videowriter.release()
    return attendence
