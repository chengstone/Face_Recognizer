import os
import sys
import shutil
import subprocess
import json
import time
import numpy as np
import cv2
import tensorflow as tf
import dlib_feature_extraction
import vgg16_feature_extraction
import facenet_feature_extraction
from face_detect_main import dlib_findFace
from face_detect_main import cv_findFace
from face_detect_main import mtcnn_findFace
from argparse import ArgumentParser
from scipy import misc
import sklearn.metrics.pairwise as pw
from moviepy.editor import VideoFileClip
from main_recognition import *


# 定义两个程序的全路径
ffprobe_path = os.path.join(os.getcwd(), "ffprobe")
ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg")


def startFindFace(dstpath, videoPath, cls):
    result_output = "result_output/"
    # 创建保存最终结果的文件夹
    checkFile(os.path.join(os.getcwd(), result_output))
    shutil.rmtree(result_output)
    os.mkdir(result_output)
    print("[1] now finding the face, this will take a while...")

    time_start = time.time()

    # 用于保存人脸图片的文件夹
    face_dst_path = os.path.join(os.getcwd(), "./face_output/")

    # 得到视频的fps
    videoCapture = cv2.VideoCapture(videoPath)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    print("fps = ", fps)

    # 识别每帧画面的人脸并保存起来
    cls.getDstFaceFileName(dstpath, face_dst_path)

    time_end = time.time()
    print("finding face using time : ", time_end - time_start, "s")
    total_time.append(time_end - time_start)

    print("[2] now running the tensorflow, this will take a while...")
    time_start = time.time()

    # 批量提取人脸特征
    cls.batch_feature_extraction(face_dst_path, batch_size=64)

    # print "file counts : ", len(dstFilenameArr)
    print("tensorflow ouput shape : ", cls.src_features.shape)
    time_end = time.time()
    print("tensorflow using time : ", time_end - time_start, "s")
    total_time.append(time_end - time_start)

    time_start = time.time()
    print("[3] now calculate similar ...")

    # 计算视频中的人脸和目标人脸的相似度
    results = cls.get_cosine_similarity_results()

    print("results = ", results)
    print("results length : ", len(results))
    print("labels length : ", len(cls.labels))
    print("dst_rects_lst length : ", len(cls.dst_rects_lst))
    print("results shape : ", np.array(results).shape)

    # bool_results中只有'真'和'假'两个值，表示每个相似度代表的人脸是不是目标人物
    # 结果是按照时间顺序保存的，当目标人物连续出现在视频中，bool_results中的结果也会是连续的
    # 在后面的算法中，会只保留第一次出现的结果，其余结果设置成'假'
    bool_results = results >= cls.threshold
    if (np.array(results).shape[1] > 1):
        print("np.array(results).shape[1] > 1")
        index_results = bool_results.take(0, 1)
        for i in range(np.array(results).shape[1]):
            index_results = np.logical_or(index_results, bool_results.take(i, 1))
    else:
        print("np.array(results).shape[1] !> 1")
        index_results = bool_results

    print(index_results)
    print(index_results.shape)

    print("==================================")

    # ****  下面算法就是用来处理目标连续出现时的情况，只保留第一次出现的结果，其余设置成False  ****
    find_flag = False
    prev_find_flag = False
    loop_index = 0
    # cls.frame_face_num是字典，key是每帧文件名，value是该帧画面的人脸数
    # 因为results是按照每帧画面的人脸顺序排列的，我们需要知道每帧画面有多少人脸才能准确处理好results
    # 循环处理每帧画面，frame_face_num_vals是每帧画面的人脸数
    for frame_face_num_vals in cls.frame_face_num.values():
        # 循环每帧画面的所有人脸
        for i in range(frame_face_num_vals):
            # 如果某个人脸是目标人物，设置成找到目标了
            if (index_results[i + loop_index] == True):
                find_flag = True

        # 判断前一帧画面是否找到了目标
        if prev_find_flag == True:
            # 如果前一帧找到目标，并且当前帧也找到了目标，说明是连续出现的
            # 那么把当前帧的结果都设置成false，只保留前一帧找到目标的结果就可以了
            if find_flag == True:
                for i in range(frame_face_num_vals):
                    index_results[i + loop_index] = False
            # 如果前一帧找到目标，但当前帧没有目标，那么设置前一帧没有找到目标继续下一帧的循环
            else:
                prev_find_flag = False
        else:
            # 如果前一帧没有找到目标，但是当前帧找到了，设置前一帧为找到目标继续下一帧的循环
            if find_flag == True:
                prev_find_flag = True
        find_flag = False
        loop_index += frame_face_num_vals

    # 经过上面算法的处理，results中值是true的帧都是每次连续出现时第一次出现的画面
    print(index_results)

    print("dst_rects_lst size : ", len(cls.dst_rects_lst))
    # 对所有帧的结果循环
    for i in range(len(index_results)):
        # 如果该帧画面找到了目标
        if (index_results[i] == True):
            # print labels[i]

            # labels中保存的是人脸文件名，比如000001_2.jpg
            # 通过文件名的方式得到帧号，比如000001，取整以后就是1
            name = int(cls.labels[i].split("_")[0])

            # 函数开始我们已经得到了视频的fps，将帧号除以fps就是该帧在视频中的秒数
            seconds = name // int(fps)
            # print seconds

            # 把秒数换算成时分秒
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)

            # 以时分秒保存的文件名
            new_file_name = str(h) + "_" + str(m) + "_" + str(s) + ".jpg"  # "{02}:{02}:{02}.jpg".format (h, m, s)
            print("find a face : " + new_file_name)

            print(i, int(cls.labels[i].split("_")[1]))
            print(dstpath + cls.labels[i].split("_")[0] + ".jpg")

            # 根据人脸文件名能够知道它所在帧的文件名，比如000001_2.jpg所在帧的文件是000001.jpg
            # 读取目标人脸所在帧的图片
            img = cv2.imread(dstpath + cls.labels[i].split("_")[0] + ".jpg")

            # 在图片上描画人脸框和相似度
            cls.draw_single_rect(img, cls.dst_rects_lst[i], (0, 255, 0))
            pen = (0, 255, 0)
            cv2.putText(img, str(np.round(results[i], 2)), (cls.dst_rects_lst[i][0], cls.dst_rects_lst[i][1] - 7),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, pen)
            # 将该图片以时分秒的形式保存到结果文件夹中
            cv2.imwrite(result_output + new_file_name, img)

    time_end = time.time()
    print(time_end - time_start, "s")
    total_time.append(time_end - time_start)
    print("-------------------------------------")
    print("processed the video time : ", total_time[0])
    print("finded face using time : ", total_time[1])
    print("tensorflow using time : ", total_time[2])
    print("calculate similar using time : ", total_time[3])

    print("total time : ", sum(total_time), "s")
    print("results picture in the ", result_output, " directory!")


# 定义获取视频信息的函数，参数是视频文件
def getVideoProbeInfo(filename):
    # 获取视频信息的命令行
    command = [ffprobe_path, "-loglevel", "quiet", "-print_format", "json", "-show_format", "-show_streams", "-i",
               filename]
    # 执行命令行
    result = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = result.stdout.read()
    # 返回json结果
    return str(out.decode('utf-8'))

# 从传入的json中返回format下的duration字段值，单位是秒
def getDuration(VIDEO_PROBE):
    data = json.loads(VIDEO_PROBE)["format"]['duration']
    return data

# 通过FFmpeg将视频的每帧保存成图片
def fullVideoProc(filename, output_dir, sec_idx, end_idx, allFrames = True, framesPerSec = 1):
    if allFrames == True:
        # 这个命令行就是上面介绍的
        command = [ffmpeg_path,"-y","-i",filename, "-ss", str(sec_idx), "-t", str(end_idx), "-q:v", "2", "-f",
                   "image2",output_dir+"%6d.jpg"]
    else:
        # 这个命令行多了一个-r，传入的framesPerSec值是1，目的是每秒只取一帧画面，可以加速处理，但是由于抛弃了很多帧画面，结果会有遗漏
        command = [ffmpeg_path, "-y", "-i", filename, "-ss", str(sec_idx), "-t", str(end_idx), "-r", str(framesPerSec), "-q:v", "2", "-f",
                   "image2", output_dir + "%6d.jpg"]
    # 执行命令行
    result = subprocess.Popen(command,shell=False,stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    out = result.stdout.read()


def checkFile(filepath):
    path = ''
    for field in filepath.split('/'):
        if len(field) > 0:
            path = path + '/' + field
            #print path, os.path.exists(path)
            if field == filepath.split('/')[-1]:
                # print path, path.find('.')
                if path.find('.') != -1:
                    if os.path.exists(path) == False:
                        os.mknod(path)
                elif os.path.exists(path) == False:
                    # print path
                    os.mkdir(path)
            elif os.path.exists(path) == False:
                os.mkdir(path)



# 开始跟之前一样，同样的参数
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

    # 参数src指定要解析的视频
    videoPath = options.src  # the image/video name you want to detect
    # 目标人物的图片
    target_arg = options.target

    total_time = []

    # 先获取视频文件的信息，json格式
    VIDEO_PROBE = getVideoProbeInfo(videoPath)
    # 得到视频文件的时长，秒为单位
    sec = float(getDuration(VIDEO_PROBE))

    print(sec)

    dstpath = os.path.join(os.getcwd(), "target_output/")

    # 接下来创建target_output文件夹，用来保存视频中的每帧图片
    checkFile(dstpath)
    shutil.rmtree(dstpath)
    os.mkdir(dstpath)

    print("[0] now processing the video, this will take a several minutes...")
    time_start = time.time()

    # 通过FFmpeg将视频每帧解出来
    fullVideoProc(videoPath, dstpath, 0, sec, True)

    time_end = time.time()
    print(time_end - time_start, "s")
    total_time.append(time_end - time_start)

    # 定义人脸识别对象，并先把目标人物的人脸特征保存起来
    cls = Recognizer(options.src, options.target, options.detect, options.feat_ext, options.threshold)

    # 开始在众多画面中找目标人物
    startFindFace(dstpath, videoPath, cls)

    print("done.")
