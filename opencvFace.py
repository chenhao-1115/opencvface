# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import os
import time
import cv2

from imutils.object_detection import non_max_suppression
video_path = 0
Classifier_path1 = './haarcascade/haarcascade_frontalface_alt.xml'
Classifier_path2 = './haarcascade/haarcascade_frontalface_alt_tree.xml'
Classifier_path3 = './haarcascade/haarcascade_frontalface_alt2.xml'
Classifier_path4 = './haarcascade/haarcascade_frontalface_default.xml'

Classifier_path = Classifier_path1
pathOut = './result/1.jpg'
face_cascade = cv2.CascadeClassifier(Classifier_path)

# np.set_printoptions(threshold=np.inf)
# pathIn = 'read.jpg'
# pathOut = './result/1.jpg'
#
# # 加载类别标签文件
# label_path='./cfg2/coco.names'
# config_path='./cfg2/yolov3-tiny.cfg'
# weights_path='./cfg2/yolov3-tiny.weights'
#
# LABELS = open(label_path).read().strip().split("\n")
# nclass = len(LABELS)
#
# np.random.seed(42)
# COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')
#
# # # 载入图片并获取其维度
# # base_path = os.path.basename(pathIn)
# # img = cv.imread(pathIn)
# # (H, W) = img.shape[:2]
#
# # 加载模型配置和权重文件
# print('从硬盘加载YOLO......')
# net = cv.dnn.readNetFromDarknet(config_path, weights_path)
#
# # 获取YOLO输出层的名字
# ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#
# def yolo_detect(pathIn='',
#                 pathOut=None,
#
#                 # label_path='./cfg2/coco.names',
#                 # config_path='./cfg2/yolov3-tiny.cfg',
#                 # weights_path='./cfg2/yolov3-tiny.weights',
#
#                 # label_path='E:/PythonProject/opencv/opencv-readVideo/cfg/coco.names',
#                 # config_path='E:/PythonProject/opencv/opencv-readVideo/cfg/yolov3_coco.cfg',
#                 # weights_path='E:/PythonProject/opencv/opencv-readVideo/cfg/yolov3_coco.weights',
#
#                 # label_path='./cfg1/coco.names',
#                 # config_path='./cfg1/yolov2-tiny.cfg',
#                 # weights_path='./cfg1/yolov2-tiny.weights',
#
#                 # label_path='./cfg3/coco.names',
#                 # config_path='./cfg3/yolov3_coco.cfg',
#                 # weights_path='./cfg3/yolov3-320.weights',
#
#                 # label_path='./cfg4/coco.names',
#                 # config_path='./cfg4/yolov2.cfg',
#                 # weights_path='./cfg4/yolov2.weights',
#
#                 # label_path='./cfg5/myv3.names',
#                 # config_path='./cfg5/my_yolov3.cfg',
#                 # weights_path='./cfg5/my_yolov3_900.weights',
#
#                 # label_path='./voccfg/voc.names',
#                 # config_path='./voccfg/yolov3.cfg',
#                 # weights_path='./voccfg/yolov3-voc_100.weights',
#
#                 # label_path='./voccfg-tiny/voc.names',
#                 # config_path='./voccfg-tiny/yolov3-tiny.cfg',
#                 # weights_path='./voccfg-tiny/yolov3-tiny_100.weights',
#                 confidence_thre=0.2,
#                 nms_thre=0.3,
#                 jpg_quality=80):
#     '''
#     pathIn：原始图片的路径
#     pathOut：结果图片的路径
#     label_path：类别标签文件的路径
#     config_path：模型配置文件的路径
#     weights_path：模型权重文件的路径
#     confidence_thre：0-1，置信度（概率/打分）阈值，即保留概率大于这个值的边界框，默认为0.5
#     nms_thre：非极大值抑制的阈值，默认为0.3
#     jpg_quality：设定输出图片的质量，范围为0到100，默认为80，越大质量越好
#     '''
#
#     # # 加载类别标签文件
#     # LABELS = open(label_path).read().strip().split("\n")
#     # # print(LABELS)
#     # nclass = len(LABELS)
#     # # print(nclass)
#     # 为每个类别的边界框随机匹配相应颜色
#     # np.random.seed(42)
#     # COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')
#     #
#     # # 载入图片并获取其维度
#     # base_path = os.path.basename(pathIn)
#     # img = cv.imread(pathIn)
#     # (H, W) = img.shape[:2]
#     #
#     # # 加载模型配置和权重文件
#     # print('从硬盘加载YOLO......')
#     # net = cv.dnn.readNetFromDarknet(config_path, weights_path)
#     #
#     # # 获取YOLO输出层的名字
#     # ln = net.getLayerNames()
#     # ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#
#     # 载入图片并获取其维度
#     base_path = os.path.basename(pathIn)
#     img = cv.imread(pathIn)
#     (H, W) = img.shape[:2]
#
#     # 将图片构建成一个blob，设置图片尺寸，然后执行一次
#     # YOLO前馈网络计算，最终获取边界框和相应概率
#     blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     # start = time.time()
#     layerOutputs = net.forward(ln)
#     # end = time.time()
#
#     # 显示预测所花费时间
#     # print('YOLO模型花费 {:.2f} 秒来预测一张图片'.format(end - start))
#
#     # 初始化边界框，置信度（概率）以及类别
#     boxes = []
#     confidences = []
#     classIDs = []
#
#     # 迭代每个输出层，总共三个
#     for output in layerOutputs:
#         # print('output:')
#         # print(output)
#         # 迭代每个检测
#         for detection in output:
#             # print(detection)
#             # 提取类别ID和置信度
#             scores = detection[5:]
#             classID = np.argmax(scores)
#             confidence = scores[classID]
#
#             # 只保留置信度大于某值的边界框
#             if confidence > confidence_thre:
#                 # 将边界框的坐标还原至与原图片相匹配，记住YOLO返回的是
#                 # 边界框的中心坐标以及边界框的宽度和高度
#                 box = detection[0:4] * np.array([W, H, W, H])
#                 (centerX, centerY, width, height) = box.astype("int")
#
#                 # 计算边界框的左上角位置
#                 x = int(centerX - (width / 2))
#                 y = int(centerY - (height / 2))
#
#                 # 更新边界框，置信度（概率）以及类别
#                 boxes.append([x, y, int(width), int(height)])
#                 confidences.append(float(confidence))
#                 # print('confidence:')
#                 # print(confidences)
#                 classIDs.append(classID)
#                 # print('classID:')
#                 # print(classIDs)
#     # 使用非极大值抑制方法抑制弱、重叠边界框
#     idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)
#
#     # 确保至少一个边界框
#     if len(idxs) > 0:
#         # 迭代每个边界框
#         for i in idxs.flatten():
#             # 提取边界框的坐标
#             (x, y) = (boxes[i][0], boxes[i][1])
#             (w, h) = (boxes[i][2], boxes[i][3])
#
#             # 绘制边界框以及在左上角添加类别标签和置信度
#             color = [int(c) for c in COLORS[classIDs[i]]]
#             cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             text = '{}: {:.3f}'.format(LABELS[classIDs[i]], confidences[i])
#             (text_w, text_h), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
#             cv.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
#             cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#
#     # 输出结果图片
#     if pathOut is None:
#         cv.imwrite('with_box_' + base_path, img, [int(cv.IMWRITE_JPEG_QUALITY), jpg_quality])
#     else:
#         cv.imwrite(pathOut, img, [int(cv.IMWRITE_JPEG_QUALITY), jpg_quality])
#

count = 0
# 定义HOG对象，采用默认参数，或者按照下面的格式自己设置
defaultHog = cv2.HOGDescriptor()
# 设置SVM分类器，用默认分类器
defaultHog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv.VideoCapture(video_path)  # 创建一个VideoCapture对象
while (True):
    ret, frame = cap.read()  # 一帧一帧读取视频
    # cv.imshow('frame', frame)  # 显示结果

    count = count+1
    if(count>20):
        count = 0

    #保存一帧
    cv.imwrite('./result/read'+str(count)+'.jpg',frame)

    if(count == 20):
        start = time.time()

        # 检测刚保存的那一帧
        faceimg = cv2.imread('./result/read'+str(count)+'.jpg')
        gray = cv2.cvtColor(faceimg, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(faceimg, (x, y), (x + w, y + w), (0, 255, 0), 2)
        cv.imwrite(pathOut,faceimg )

        # hogimg = cv2.imread('read'+str(count)+'.jpg')
        # roi = hogimg
        # cv2.imshow("roi", roi)
        # cv2.imwrite("roi.jpg", roi)
        # (rects, weights) = defaultHog.detectMultiScale(roi, winStride=(4, 4), padding=(8, 8), scale=1.05)
        # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        # pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # for (xA, yA, xB, yB) in pick:
        #     cv2.rectangle(roi, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # cv.imwrite(pathOut, roi)

        # yolo_detect('read'+str(count)+'.jpg', pathOut)
        #输出查看标记情况
        result = cv.imread(pathOut)
        cv.imshow('result', result)  # 显示结果
        end = time.time()
        # print('YOLO模型花费 {:.2f} 秒来预测一张图片'.format(end - start))
        # print('HOG模型花费 {:.2f} 秒来预测一张图片'.format(end - start))
        print('face模型花费 {:.2f} 秒来预测一张图片'.format(end - start))
    if cv.waitKey(1) & 0xFF == ord('q'):  # 按q停止
        break

cap.release()  # 释放cap,销毁窗口
cv.destroyAllWindows()