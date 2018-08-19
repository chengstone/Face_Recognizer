from scipy import misc
import numpy as np
import os
import cv2
import dlib
# 引入上一节定义的MTCNN人脸检测的函数mtcnn_findFace
from face_detect_main import mtcnn_findFace

# 特征点预测器文件和dlib人脸识别模型文件
shape_predictor_68_face_landmarks = "./shape_predictor_68_face_landmarks.dat"
shape_predictor_5_face_landmarks = "./shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"

# 转换人脸框为dlib的rectangle对象
def convert_to_rect(rect):
    return dlib.rectangle(rect[0], rect[1], rect[2], rect[3])


def feature_extraction_single(img, rect, mode="68"):

    feature = []

    show_img = img.copy()

    # 定义特征点预测器
    if mode == "68":
        sp = dlib.shape_predictor(shape_predictor_68_face_landmarks)
    else:
        sp = dlib.shape_predictor(shape_predictor_5_face_landmarks)

    # 定义dlib人脸识别模型
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    # 计算每张人脸的特征点
    shape = sp(img, convert_to_rect(rect))

    # 把特征点画出来，检验程序是否正确
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        cv2.circle(show_img, pt_pos, 2, (0, 255, 0), 1)

    # 将原图像和特征点传入人脸识别模型，得到人脸特征
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    feature.append(face_descriptor)

    feature = np.array(feature)
    # 结果保存到本地文件
    misc.imsave("./dlib_feature_extraction_{}.png".format(mode), show_img)

    # 返回人脸特征
    return feature


# 提取人脸特征
def feature_extraction(img, rects, mode="68"):

    feature = []

    if len(rects) > 0:

        show_img = img.copy()

        # 定义特征点预测器
        if mode == "68":
            sp = dlib.shape_predictor(shape_predictor_68_face_landmarks)
        else:
            sp = dlib.shape_predictor(shape_predictor_5_face_landmarks)

        # 定义dlib人脸识别模型
        facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        for rect in rects:
            # 计算每张人脸的特征点
            shape = sp(img, convert_to_rect(rect))

            # 把特征点画出来，检验程序是否正确
            for pt in shape.parts():
                pt_pos = (pt.x, pt.y)
                cv2.circle(show_img, pt_pos, 2, (0, 255, 0), 1)

            # 将原图像和特征点传入人脸识别模型，得到人脸特征
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            feature.append(face_descriptor)

        feature = np.array(feature)
        # 结果保存到本地文件
        misc.imsave("./dlib_feature_extraction_{}.png".format(mode), show_img)

    # 返回人脸特征
    return feature

if __name__ == '__main__':
    # 先通过MTCNN取得人脸框位置
    image_path = "./frame_tmp.jpg"
    img = misc.imread(os.path.expanduser(image_path), mode='RGB')
    images, rects = mtcnn_findFace(img)

    img = misc.imread(os.path.expanduser(image_path), mode='RGB')
    feature_extraction(img, rects, "68")

    img = misc.imread(os.path.expanduser(image_path), mode='RGB')
    feature_extraction(img, rects, "5")
