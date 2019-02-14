import facenet
import tensorflow as tf
from scipy import misc
import numpy as np
import os
import cv2
# 引入上一节定义的MTCNN人脸检测的函数mtcnn_findFace
from face_detect_main import mtcnn_findFace

facenet_graph = tf.Graph()

with facenet_graph.as_default():
    facenet_sess = tf.compat.v1.Session(graph=facenet_graph)
    facenet_sess.run(tf.compat.v1.global_variables_initializer())

    with facenet_sess.as_default():
        # 加载模型
        facenet.load_model('./model/20180402-114759/')

        # 获得输入和输出的张量，images_placeholder是输入图像，embeddings是输出的特征
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

# 传入findFace返回的人脸图像
def feature_extraction(tmp_img):
    # 图像预处理
    prewhitened = facenet.prewhiten(tmp_img)

    if prewhitened.ndim == 3:
        prewhitened = np.expand_dims(prewhitened, 0)

    with facenet_graph.as_default():

        # 执行前向传播计算人脸特征
        feed_dict = {images_placeholder: prewhitened, phase_train_placeholder: False}
        emb = facenet_sess.run(embeddings, feed_dict=feed_dict)

    # 返回人脸特征
    return emb

if __name__ == '__main__':
    image_path = "./frame_tmp.jpg"
    img = misc.imread(os.path.expanduser(image_path), mode='RGB')
    images, rects = mtcnn_findFace(img)
    feature_extraction(images)
