# Face_Recognizer
face recognition application

本项目是对之前的FindFaceInVideo项目的重构，当时写的代码实在是太烂了。
本次包含了OpenCV、dlib和MTCNN（facenet实现）三种人脸检测方法，包含dlib、facenet和VGG16三种人脸特征提取方法。

项目中用到的一些模型需要你下载放到本项目根目录下，因为太大就没有一起上传：

1、facenet的模型文件，请到https://github.com/davidsandberg/facenet地址去下载。
modelname是20180402-114759，下载链接是：https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-
放到本项目的model文件夹下。

2、vgg16的模型文件，请到https://github.com/machrisaa/tensorflow-vgg地址去下载。
vgg16.npy下载链接是：https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
放到本项目的tensorflow_vgg文件夹下。

3、ffmpeg和ffprobe，请到http://ffmpeg.org/download.html地址去下载。
放到本项目根目录下。

4、dlib的模型文件，请到http://dlib.net/files/地址去下载。
需要下载以下四个文件：
shape_predictor_5_face_landmarks.dat.bz2
shape_predictor_68_face_landmarks.dat.bz2  
dlib_face_recognition_resnet_model_v1.dat.bz2    
mmod_human_face_detector.dat.bz2  
解压并放到本项目根目录下

文件结构：
align：拷贝自facenet的facenet/src/align/文件夹
model：用来保存facenet的模型文件
tensorflow_vgg：https://github.com/machrisaa/tensorflow-vgg项目的源代码
result_output：保存本项目最终的输出结果
chengshd：我的测试数据，里面是我的相关照片和视频
vgg16_feature_extraction.py：使用vgg16提取特征
facenet_feature_extraction.py：使用facenet提取特征
dlib_feature_extraction.py：使用dlib提取特征
face_detect_main.py：包含OpenCV、dlib和MTCNN三种人脸检测方法
main_recognition.py：人脸识别类的代码，可以对照片和视频做人脸识别，并输出照片或视频
action.py：另外一种应用场景的实现。给定视频找目标人脸，并输出目标在首次视频出现时的照片。
