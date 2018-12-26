# encoding:utf-8
import numpy as np
import cv2
import sys
import os

FREQ_DIV = 5  # 保存图片的间隔
RESIZE_FACTOR = 4  # 图像尺寸缩小的倍数
NUM_TRAINING = 100  # 用于训练人脸的数据集的大小


class TrainEigenFaces:
    def __init__(self):  # 初始化
        cascPath = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)  # 加载人脸检测文件
        self.face_dir = 'face_data'  # 训练集的文件路径
        self.face_name = sys.argv[1]  # gui界面输入的识别的人脸的名字
        self.path = os.path.join(self.face_dir, self.face_name)  # 判断文件是否存在
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        self.model = cv2.face.EigenFaceRecognizer_create()  # 读取特征脸
        self.count_captures = 0  # 程序读取的图像数
        self.count_timer = 0  # 控制提取视频图片的间隔

    # 从计算机摄像头录取每一帧图像，进行处理，最后按下’q‘关闭视频
    def capture_training_images(self):
        video_capture = cv2.VideoCapture(0)
        while True:
            self.count_timer += 1
            ret, frame = video_capture.read()
            inImg = np.array(frame)
            outImg = self.process_image(inImg)
            cv2.imshow("Reading your face", outImg)

            # 按下键盘键－'q'关闭视频流
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_capture.release()
                cv2.destroyAllWindows()
                return

    def process_image(self, inImg):
        frame = cv2.flip(inImg, 1)  # 水平翻转图像
        resized_width, resized_height = (112, 92)
        if self.count_captures < NUM_TRAINING:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 图像均衡增强处理
            qgray = cv2.equalizeHist(gray)
            img_row = gray.shape[0]  # 图像矩阵的行数==height
            img_col = gray.shape[1]  # 图像矩阵的列数==weigh

            gray_resized = cv2.resize(gray, (img_col // RESIZE_FACTOR, img_row // RESIZE_FACTOR))  # 将图片缩小

            # 人脸检测
            faces = self.face_cascade.detectMultiScale(
                gray_resized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                areas = []
                for (x, y, w, h) in faces:
                    areas.append(w * h)
                max_area, idx = max([val, idx] for idx, val in enumerate(areas))  # val表示矩形面积的大小
                face_sel = faces[idx]  # 从检测得到的人脸矩形中获取面积最大的矩形框

                # 还原坐标，用于原始图像中抠出人脸
                x = face_sel[0] * RESIZE_FACTOR
                y = face_sel[1] * RESIZE_FACTOR
                w = face_sel[2] * RESIZE_FACTOR
                h = face_sel[3] * RESIZE_FACTOR

                # 原始图像中抠出人脸
                face = gray[y:y + h, x:x + w]  # 左上右下

                # resized
                face_resized = cv2.resize(face, (resized_width, resized_height))

                # 提取图片路径中的图片的序号序列，并在序列最后加0，保证训练图片重新提取重新计数
                img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir(self.path) if fn[0] != '.'] + [0])[-1] + 1

                if self.count_timer % FREQ_DIV == 0:
                    cv2.imwrite('%s/%s.png' % (self.path, img_no), face_resized)
                    self.count_captures += 1
                    print("INFO--成功提取%s张图片....." % self.count_captures)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, self.face_name, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 23, 0))
        elif self.count_captures == NUM_TRAINING:
            print("INFO--%s张图片提取完成，请按下'q'结束程序" % NUM_TRAINING)
            self.count_captures += 1

        return frame

    def eigen_train_data(self):
        imgs = []
        tags = []
        index = 0

        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                img_path = os.path.join(self.face_dir, subdir)  # 提取某个人的所有训练图片的文件名
                for fn in os.listdir(img_path):
                    path = img_path + '/' + fn
                    tag = index
                    imgs.append(cv2.imread(path, 0))  # 提取某个人的图片到一个图像集中
                    tags.append(int(tag))
                index += 1
        (imgs, tags) = [np.array(item) for item in [imgs, tags]]

        # 训练模型
        self.model.train(imgs, tags)
        self.model.save('eigen_trained_data.xml')
        print("INFO--模型保存成功")
        return


if __name__ == '__main__':
    trainer = TrainEigenFaces()
    trainer.capture_training_images()
    trainer.eigen_train_data()
    print("INFO--人脸识别训练成功")
