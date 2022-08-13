# Smart-Attendance-System
## Build Custom VGGFace Model

We apply some changes to the main VGGFace model. The top layer of the model is Softmax layer which trained to predict 9131 different faces in our environment the number of students changes over time, so we change the model architecture during the interface with the model and training the model online continuously when every new student came to log in a course or semester.


![image](https://user-images.githubusercontent.com/43546116/184511320-ab793d1c-ab4d-470c-9ea7-a8a616706443.png)
![image](https://user-images.githubusercontent.com/43546116/184511326-3a3c6707-fb09-4cc4-b00a-5e7605b78ef2.png)
