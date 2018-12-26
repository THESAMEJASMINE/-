# encoding:utf-8

from tkinter import *
import os

root = Tk(className='人脸识别')

svalue = StringVar()
root.geometry('400x200')
notice = Label(root, text="请输入名字", font='Helvetica -16 bold')
notice.pack()
w = Entry(root, textvariable=svalue)
w.pack()


def detecte_eigen_btn_load():
    os.system('python FaceDetecte.py')


def train_eigen_btn_load():
    name = svalue.get()
    os.system('python FaceRecog_Eigen_train.py %s' % name)


def recog_eigen_btn_load():
    os.system('python FaceRecog_Eigen_recog.py')


deteE_btn = Button(root, text="人脸检测", font='Helvetica -16 bold', command=detecte_eigen_btn_load)
deteE_btn.pack()

trainE_btn = Button(root, text="录入脸部信息", font='Helvetica -16 bold', command=train_eigen_btn_load)
trainE_btn.pack()

recogE_btn = Button(root, text="识别人脸", font='Helvetica -16 bold', command=recog_eigen_btn_load)
recogE_btn.pack()

root.mainloop()
