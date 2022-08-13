# Smart-Attendance-System
### Build Custom VGGFace Model

We apply some changes to the main VGGFace model. The top layer of the model is Softmax layer which trained to predict 9131 different faces in our environment the number of students changes over time, so we change the model architecture during the interface with the model and training the model online continuously when every new student came to log in a course or semester.

We drop the last layer of the VGGface model and add another two dense layers of 512 and 128 neurons every one of them is followed by Dropout layer with 40% and 10% probability and the last layer is a Softmax layer with the number of students. Every time we add a new student to the database, we build a new model with the same architecture described above but with new Softmax layer with the new number of students and we hard copy the old weights to the new model and pretraining the new custom model from the old one.

### To Run the Program
python –video ‘video path’ –show_video “True or False to see the pipeline result while excuting” –cource_name “cource name” –output “output file path”.
### Some Output Examples
![image](https://user-images.githubusercontent.com/43546116/184511320-ab793d1c-ab4d-470c-9ea7-a8a616706443.png)
![image](https://user-images.githubusercontent.com/43546116/184511326-3a3c6707-fb09-4cc4-b00a-5e7605b78ef2.png)
