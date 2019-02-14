# 首先将facenet项目中的align文件夹拷贝到我们的代码同级目录，引入align.detect_face
import align.detect_face
import tensorflow as tf
from scipy import misc
import numpy as np
import os
import cv2
import dlib

######################  dlib 人脸检测 ######################

# dlib官方提供的人脸检测器模型
mmod_human_face_detector = "./mmod_human_face_detector.dat"


# 用来将人脸检测器返回的人脸框位置转换成[x1, y1, x2, y2]的形式
def convert_rect(rect):
    return [rect.left(), rect.top(), rect.right(), rect.bottom()]


# 人脸检测主函数，mode用来指定使用普通检测器还是CNN检测器
def dlib_findFace(img, mode="cnn", image_size=160):

    if mode == "cnn":
        # 传入人脸检测器模型，获得CNN检测器
        cnn_face_detector = dlib.cnn_face_detection_model_v1(mmod_human_face_detector)
        # 检测人脸，返回人脸对象
        faces = cnn_face_detector(img, 1)
        rects = []
        # 转换人脸对象中的人脸框并保存
        for face in faces:
            rects.append(convert_rect(face.rect))
    else:
        # 获得人脸检测器
        face_detector = dlib.get_frontal_face_detector()
        # 检测人脸，返回人脸框
        boxs = face_detector(img, 1)
        rects = []
        # 转换人脸框并保存
        for rect in boxs:
            rects.append(convert_rect(rect))

    if len(rects) != 0:
        img_list = []
        # 将框出的人脸保存起来
        for rect in rects:
            vis = img[rect[1]:rect[3], rect[0]:rect[2], :]
            aligned = misc.imresize(vis, (image_size, image_size), interp='bilinear')
            img_list.append(aligned)
        # 在图像上画出检测到的每张人脸
        draw_rects(img, rects, (0, 255, 0))

        # 结果保存到本地文件
        misc.imsave("./dlib_face_detect_{}.png".format(mode), img)

        # 返回检测到的人脸和人脸框
        images = np.stack(img_list)
        return images, rects
    else:
        return [], []

######################  OpenCV 人脸检测 ######################

# 四个人脸检测器的文件名
haarcascade_frontalface_default = "haarcascade_frontalface_default.xml"      # 人脸检测器（默认）
haarcascade_frontalface_alt2 = "haarcascade_frontalface_alt2.xml"            # 人脸检测器（快速的Haar）
haarcascade_frontalface_alt_tree = "haarcascade_frontalface_alt_tree.xml"    # 人脸检测器（Tree）
haarcascade_frontalface_alt = "haarcascade_frontalface_alt.xml"              # 人脸检测器（Haar_1）

# 人脸检测器所在路径
# CAS_PATH = "/Applications/anaconda/pkgs/libopencv-3.4.1-he076b03_1/share/OpenCV/haarcascades/"
CAS_PATH = "/anaconda3/envs/tf2/share/OpenCV/haarcascades/"

# 检测函数，传入灰度图和级联分类器
def detect(img, cascade):
    # 调用级联分类器的人脸检测函数，返回人脸框
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

# 在图像上描画出人脸框
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

# 人脸检测主函数
def cv_findFace(img, image_size=160):

    # 将图像转成灰度图，并做直方图均衡化提高图像质量
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    # 使用人脸检测器创建级联分类器
    cascade_fn = os.path.join(CAS_PATH, haarcascade_frontalface_alt)
    cascade = cv2.CascadeClassifier(cascade_fn)
    # 使用级联分类器检测人脸
    rects = detect(gray, cascade)

    if len(rects) != 0:
        img_list = []
        # 将框出的人脸保存起来
        for rect in rects:
            vis = img[rect[1]:rect[3], rect[0]:rect[2], :]
            aligned = misc.imresize(vis, (image_size, image_size), interp='bilinear')
            img_list.append(aligned)
        # 在图像上画出检测到的每张人脸
        draw_rects(img, rects, (0, 255, 0))

        # 结果保存到本地文件
        misc.imsave("./cv_face_detect.png", img)

        # 返回检测到的人脸和人脸框
        images = np.stack(img_list)
        return images, rects
    else:
        return [], []


######################  MTCNN 人脸检测 ######################

# 首先调用create_mtcnn创建MTCNN的三个网络
mtcnn_graph = tf.Graph()
with mtcnn_graph.as_default():
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1)
    mtcnn_sess = tf.compat.v1.Session(graph=mtcnn_graph,
                            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    mtcnn_sess.run(tf.compat.v1.global_variables_initializer())
    with mtcnn_sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(mtcnn_sess, None)


# 画矩形框
def draw_single_rect(img, rect, color):
    cv2.rectangle(img, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), color, 2)


# 人脸检测
def mtcnn_findFace(img, image_size=160, margin=44):
    minsize = 20  # 人脸的最小尺寸
    threshold = [0.6, 0.7, 0.7]  # 三个网络判断是否是人脸的阈值
    factor = 0.709  # 缩放因子

    # 调用detect_face得到人脸框
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    img_list = []
    rects = []

    for i in range(len(bounding_boxes)):
        det = np.squeeze(bounding_boxes[i, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])

        rects.append([bb[0], bb[1], bb[2], bb[3]])
        # 将检测到的人脸框抠出来并保存
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        img_list.append(aligned)

        # 将检测到的人脸框描画出来
        draw_single_rect(img, bb, (0, 255, 0))

    # 结果保存到本地文件
    misc.imsave("./mtcnn_face_detect.png", img)

    # 返回检测到的人脸和人脸框
    images = np.stack(img_list)
    return images, rects


if __name__ == '__main__':
    image_path = "./dest.jpg"
    img = misc.imread(os.path.expanduser(image_path), mode='RGB')
    mtcnn_findFace(img)

    img = misc.imread(os.path.expanduser(image_path), mode='RGB')
    cv_findFace(img)

    img = misc.imread(os.path.expanduser(image_path), mode='RGB')
    dlib_findFace(img, mode="cnn", image_size=160)

    img = misc.imread(os.path.expanduser(image_path), mode='RGB')
    dlib_findFace(img, mode="normal", image_size=160)
