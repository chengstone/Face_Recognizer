# 引入前几节完成的特征提取和人脸检测函数
import dlib_feature_extraction
import vgg16_feature_extraction
import facenet_feature_extraction
from face_detect_main import dlib_findFace
from face_detect_main import cv_findFace
from face_detect_main import mtcnn_findFace
from argparse import ArgumentParser
from scipy import misc
import numpy as np
import os
import sklearn.metrics.pairwise as pw
import cv2
from moviepy.editor import VideoFileClip
from action import *


# 定义人脸识别类
class Recognizer():
    # 初始化函数，传入的是两张图片的路径，人脸检测类型，特征提取类型和阈值
    # 函数主要目的是提取目标人物的人脸特征
    def __init__(self, src_path, target_path, detect, feat_ext, threshold):
        self.src_path = src_path
        self.detect = detect
        self.feat_ext = feat_ext
        self.threshold = threshold

        self.filenames = []
        self.dst_rects_lst = []
        self.frame_face_num = {}

        # 先检测目标图像，提取出人脸特征
        img = misc.imread(os.path.expanduser(target_path), mode='RGB')
        self.target_images, self.target_rects = self.findFace(detect, img.copy())
        self.target_features = self.feature_extraction(feat_ext, self.target_images, self.target_rects, img.copy())

    # 人脸检测函数，根据传入的检测类型选择不同的检测方法
    def findFace(self, detect, img):
        if detect == 'mtcnn':
            return mtcnn_findFace(img)
        elif detect == 'cv':
            return cv_findFace(img)
        elif detect == 'dlib_cnn':
            return dlib_findFace(img, mode="cnn", image_size=160)
        elif detect == 'dlib':
            return dlib_findFace(img, mode="normal", image_size=160)
        else:
            return [], []

    def batch_feature_extraction(self, face_dst_path, batch_size=64):
        batch = []
        self.labels = []
        self.src_features = None

        dlib_src_img = []
        dlib_src_rect = []

        # 所有人脸文件的循环
        for ii, file in enumerate(self.filenames, 1):
            # 保存人脸文件名，不包含后缀，例：000001_2
            self.labels.append(file.split("/")[-1].split(".")[0])

            if self.feat_ext == 'dlib_68':
                # dlib比较特殊，需要提供人脸所在帧的图片和人脸框
                # 读取人脸所在帧的图片并保存到dlib_src_img中，例：000001.jpg
                dlib_img = misc.imread(os.path.expanduser(os.path.join(self.dstpath, file.split("_")[0] + ".jpg")),
                                       mode='RGB')
                dlib_src_img.append(dlib_img)
                # 保存人脸框
                dlib_src_rect.append(self.dst_rects_lst[ii - 1])
            else:
                # 读取人脸图片并保存到batch中
                img = misc.imread(os.path.expanduser(os.path.join(face_dst_path, file)), mode='RGB')
                if self.feat_ext == 'facenet':
                    batch.append(img.reshape((160, 160, 3)))
                else:
                    img = misc.imresize(img, (224, 224, 3), interp='bilinear')
                    batch.append(img.reshape((1, 224, 224, 3)))

            # 如果读取的人脸数达到一个batch批次了，或者所有人脸都加载进来了，开始批量提取人脸特征
            if ii % batch_size == 0 or ii == len(self.filenames):

                if self.feat_ext == 'facenet':
                    images = np.stack(batch)
                    codes_batch = facenet_feature_extraction.feature_extraction(images)
                elif self.feat_ext == 'vgg16':
                    images = np.concatenate(batch)
                    codes_batch = vgg16_feature_extraction.feature_extraction(images)
                elif self.feat_ext == 'dlib_68':
                    images = np.stack(dlib_src_img)  # concatenate
                    if images.ndim > 3:
                        codes_batch = []
                        for i in range(images.shape[0]):
                            feature = dlib_feature_extraction.feature_extraction_single(images[i], dlib_src_rect[i],
                                                                                        "68")
                            codes_batch.append(feature)
                    else:
                        codes_batch = dlib_feature_extraction.feature_extraction_single(images, dlib_src_rect[0], "68")

                # 保存人脸特征
                if self.src_features is None:
                    self.src_features = codes_batch
                else:
                    self.src_features = np.concatenate((self.src_features, codes_batch))

                # 重置变量，开始下一批次的循环
                batch = []
                dlib_src_img = []
                dlib_src_rect = []
                print('{} images processed'.format(ii))

    # 特征提取函数，根据传入的特征提取类型选择不同的特征提取方法
    def feature_extraction(self, feat_ext, images, rects, img):
        if feat_ext == 'facenet':
            return facenet_feature_extraction.feature_extraction(images)
        elif feat_ext == 'vgg16':
            return vgg16_feature_extraction.feature_extraction(images)
        elif feat_ext == 'dlib_68':
            return dlib_feature_extraction.feature_extraction(img, rects, "68")
        else:
            return []

    # 余弦相似度函数，计算两个特征间的相似度
    def cosine_similarity(self, src_feature, target_feature):
        if len(src_feature) == 0 or len(target_feature) == 0:
            return np.empty((0))

        predicts = pw.cosine_similarity(src_feature, target_feature)
        return predicts

    # 欧氏距离函数，计算两个特征间的欧氏距离
    def euclidean_distance(self, src_feature, target_feature):
        if len(src_feature) == 0 or len(target_feature) == 0:
            return np.empty((0))

        return np.linalg.norm(src_feature - target_feature, axis=1)

    # 在图像上描画矩形框
    def draw_single_rect(self, img, rect, color):
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)

    def get_cosine_similarity_results(self):
        # 因为目标人脸有可能是一个，在维度上跟批量人脸不同，比如目标特征维度可能是(1, 128)，
        # 而批量人脸特征的维度可能是(batch_size, 1, 128)
        # 所以需要对批量特征reshape一下成(batch_size, 128)
        if self.src_features.ndim != self.target_features.ndim:
            self.src_features = np.reshape(self.src_features,
                                           (-1, self.target_features.shape[self.target_features.ndim - 1]))
        # 得到余弦相似度
        return self.cosine_similarity(self.src_features, self.target_features)

    # 人脸识别主函数，从image图像中识别目标人物
    def process(self, image):
        # 先将图片中的所有人脸特征提取出来
        self.src_images, self.src_rects = self.findFace(self.detect, image.copy())
        self.src_features = self.feature_extraction(self.feat_ext, self.src_images, self.src_rects, image.copy())

        # 计算特征间的欧氏距离
        distances = self.euclidean_distance(self.src_features, self.target_features)
        # 计算特征间的余弦相似度
        cosine_distances = self.cosine_similarity(self.src_features, self.target_features)

        # 如果没有找到，退出函数
        if len(cosine_distances) == 0 or len(distances) == 0:
            return image

        # 得到相似度最大的下标
        index_x, index_y = np.where(cosine_distances == np.max(cosine_distances))
        # 循环描画人脸框、欧氏距离和相似度
        for i in range(len(cosine_distances)):
            # 如果当前下标是相似度最大值的下标，并且相似度大于阈值，说明找到目标人物了，用绿色表示
            if i == index_x and cosine_distances[i] >= self.threshold:
                pen = (0, 255, 0)
            else:
                # 否则不是目标人物，用红色表示
                pen = (255, 0, 0)

            # 画出矩形框、欧氏距离和相似度
            self.draw_single_rect(image, self.src_rects[i], pen)
            cv2.putText(image, str(np.round(cosine_distances[i], 2)),
                        (self.src_rects[i][0], self.src_rects[i][1] - 7),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, pen)  # "similarity : " +
            cv2.putText(image, str(round(distances[i], 2)),
                        (self.src_rects[i][0], self.src_rects[i][1] - 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, pen)  # "distance : " +

        # 结果保存到本地文件
        misc.imsave("./recognition_result.png", image)
        return image

        # 在每帧画面中找出人脸，并保存起来
    def getDstFaceFileName(self, dstpath, face_dst_path):
        # 创建用于保存人脸的文件夹
        checkFile(face_dst_path)
        shutil.rmtree(face_dst_path)
        os.mkdir(face_dst_path)
        self.dstpath = dstpath

        filenames = os.listdir(dstpath)
        # 每帧画面的循环
        for ii, file in enumerate(filenames, 1):
            image_path = os.path.join(dstpath, file)
            # 读取每帧画面
            image = misc.imread(image_path, mode='RGB')
            # 识别出人脸，得到人脸图像和人脸框
            self.src_images, self.src_rects = self.findFace(self.detect, image.copy())
            # 保存每帧画面的人脸数
            self.frame_face_num[image_path.split("/")[-1].split(".")[0]] = len(self.src_rects)
            # 调用下面函数保存每张人脸
            self.getDstFaceFileName_proc(image_path, face_dst_path)

    # 循环保存每张人脸
    def getDstFaceFileName_proc(self, image_path, face_dst_path):
        # 每张人脸的循环
        for idx, rect in enumerate(self.src_rects):
            vis = self.src_images[idx]
            # 保存人脸，文件名命名规则是帧号加人脸序号，比如000001_2.jpg
            cv2.imwrite(
                os.path.join(face_dst_path, image_path.split("/")[-1].split(".")[0] + "_" + str(idx) + ".jpg"), vis)
            # 保存人脸文件名
            self.filenames.append(image_path.split("/")[-1].split(".")[0] + "_" + str(idx) + ".jpg")
            # 保存对应的人脸框
            self.dst_rects_lst.append(rect)


if __name__ == '__main__':
    # 设定参数，一共五组参数
    # detect：表示人脸检测类型。一共四种类型，'mtcnn', 'cv', 'dlib_cnn', 'dlib'。
    # feat_ext：表示特征提取类型。一共三种类型，'facenet', 'vgg16', 'dlib_68'。
    # src：源图像路径，从该图像中识别目标人物。
    # target：目标图像路径，指定要找的人是谁。
    # threshold：决定是否找到人的阈值。相似度大于等于该值，则说明成功找到目标。
    parser = ArgumentParser()
    parser.add_argument('--detect', default='dlib', choices=['mtcnn', 'cv', 'dlib_cnn', 'dlib'], type=str,
                        help='mtcnn, cv, dlib_cnn, dlib')
    parser.add_argument('--feat_ext', default='dlib_68', choices=['facenet', 'vgg16', 'dlib_68'], type=str,
                        help='facenet, vgg16, dlib_68')
    parser.add_argument('--src', dest='src', help='image/video path', required=True)
    parser.add_argument('--target', default="", dest='target', help='image/video path', required=True)
    parser.add_argument('--threshold', type=float, default=0.8,
                        dest='threshold', help='the videos and pictures threshold',
                        metavar='THRESHOLD')

    options = parser.parse_args()

    # 定义人脸识别对象
    cls = Recognizer(options.src, options.target, options.detect, options.feat_ext, options.threshold)

    # 判断是否从图片中找人
    if options.src.split('.')[-1] in ['jpg', 'JPG', 'jpeg', 'bmp', 'png']:
        src_img = misc.imread(os.path.expanduser(options.src), mode='RGB')
        # 开始人脸识别
        cls.process(src_img)
    # 判断是否从视频中找人
    elif options.src.split('.')[-1] in \
            ['mov', 'MOV', 'rm', 'RM', \
             'rmvb', 'RMVB', 'mp4', 'MP4', \
             'avi', 'AVI', 'wmv', 'WMV', \
             '3gp', '3GP', 'mpeg', 'MPEG', \
             'mkv', 'MKV']:
        # 输出文件路径
        outpath = './result_out.mp4'
        # 定义VideoFileClip对象，传入视频文件路径
        video_clip = VideoFileClip(options.src)
        # 设置回调函数，用来接收每帧图像
        video_out_clip = video_clip.fl_image(cls.process)
        # 开始视频处理，并将结果保存到输出文件
        video_out_clip.write_videofile(outpath, audio=False)

