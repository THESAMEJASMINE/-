# encoding:utf-8
# 人脸检测
import numpy as np
import cv2 as cv

# 实时视频,调用摄像头
cv.namedWindow("Face Detected")
cap = cv.VideoCapture(0)
success, frame = cap.read()

# 　加载opencv识别器
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# 定义人脸矩形
[x, y, w, h] = [0, 0, 0, 0]

# 每一帧的图像进行处理
while success:
    success, frame = cap.read()
    size = frame.shape[:2]  # 读入的视频矩阵大小
    image = np.zeros(size, dtype=np.float16)  # 初始化一个图像矩阵
    image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 直方图均衡
    image = cv.equalizeHist(image)
    im_h, im_w = size
    minSize_1 = (im_w // 10, im_h // 10)
    faceRects = face_cascade.detectMultiScale(image, 1.05, 2, cv.CASCADE_SCALE_IMAGE, minSize_1)
    only = 0
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect

            face_im = np.zeros([w, h], dtype=np.float16)  # 初始化一个眼睛图像矩阵
            temp_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            face_im = temp_image[y:y + h, x:x + w]  # 从图片中抠出脸部
            eyeCicle = eyes_cascade.detectMultiScale(face_im, 1.05, 2, cv.CASCADE_SCALE_IMAGE, (w // 10, h // 10))
            if len(eyeCicle) == 2:
                x1, y1, w1, h1 = eyeCicle[0]
                x2, y2, w2, h2 = eyeCicle[1]
                point_1 = [(2 * (x + x1) + w1) // 2, (2 * (y + y1) + h1) // 2]
                point_2 = [(2 * (x + x2) + w2) // 2, (2 * (y + y2) + h2) // 2]
                r1 = w1 // 2
                r2 = w2 // 2
                cv.circle(frame, (point_1[0], point_1[1]), r1, (255, 0, 255), 2)
                cv.circle(frame, (point_2[0], point_2[1]), r2, (255, 0, 255), 2)
                if (x2 > x1):
                    cv.line(frame, (point_1[0] + r1, point_1[1]), (point_2[0] - r2, point_2[1]), (255, 0, 255), 2)
                else:
                    cv.line(frame, (point_2[0] + r2, point_2[1]), (point_1[0] - r1, point_1[1]), (255, 0, 255), 2)
    cv.rectangle(frame, (x, y), (x + w, y + h), [255, 255, 0], 2)
    cv.imshow("Face Detected", frame)
    if only == 0:
        cv.imwrite('out/检测.png', frame)
        only = 1
    # key = cv.waitKey(5)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
cv.destroyWindow("Face Detection System")
