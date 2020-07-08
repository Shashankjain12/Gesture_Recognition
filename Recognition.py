#importing the libraries

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import keras
import imutils
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from textgenrnn import textgenrnn
from multiprocessing import Process,Queue
from multiprocessing import Pool

#Loading the model

model=load_model("./models/hands_sign_binary_final.hdf5")
txt=textgenrnn()

#Detection and Background removal from the frame after loading the model

#Edge Detection 
def edgedetect( channel):
    sobelX=cv2.Sobel(channel,cv2.CV_16S,1,0)
    sobelY=cv2.Sobel(channel,cv2.CV_16S,0,1)
    sobel=np.hypot(sobelX,sobelY)
    sobel[sobel>255]=255
    return sobel


cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 100
learningRate = 0
var_list=[]
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

#Print Threshold value function
def printThreshold(thr):
    print("! Changed threshold to "+str(thr))

#Background Removal
def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

#Detection of hand gestures

def detect(gray,frame,thresh):
    cv2.rectangle(frame,(520,100),(790,500),(255,0,0),0)
    lab={'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'k': 9, 'l': 10, 'm': 11, 'n': 12, 'o': 13, 'p': 14,
     'q': 15, 'r': 16, 's': 17, 't': 18, 'u': 19, 'v': 20, 'w': 21, 'x': 22, 'y': 23}
    global var_list    
    image=cv2.resize(thresh,(28,28))
    image=image.astype("float")/255.0
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    pred=model.predict(image)
    prob=np.max(pred)
    labels=np.argmax(pred)
    for i,j in lab.items():
        if j==labels:
            labels=i    
    if prob>0.8:
        cv2.putText(frame,labels,(500,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2,cv2.LINE_AA,False)
        var_list.append(labels)
        word_gen(var_list)
        print("labels:",labels,"probability:",prob)
    return frame

def word_gen(var_list):
    if len(var_list)>0:
        a=txt.generate(n=1,return_as_list=True,max_gen_length=4,prefix=var_list[-1])
        cv2.putText(frame,a[-1],(500,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2,cv2.LINE_AA,False)


#UI Creation
if __name__=="__main__":
    # OpenCv for Video Capturing
    cap=cv2.VideoCapture(-1)
    cap.set(10,200)
    cv2.namedWindow('trackbar')
    cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)
    count=0
    while True:
        ret,frame=cap.read()
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame=cv2.resize(frame,(940,640),interpolation=cv2.INTER_LINEAR)
        frame=cv2.flip(frame,1)
        frame1=frame[100:500,520:790]
        if isBgCaptured == 1:
            # this part wont run until background captured
            img = removeBG(frame1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('roi', thresh)
            canvas=detect(gray,frame,thresh)
            cv2.imshow("image",canvas)

        k = cv2.waitKey(10)

        if k == ord('q'):
            # press ESC to exit
            cap.release()
            cv2.destroyAllWindows()
            break

        elif k == ord('b'):  
            # press 'b' to capture the background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = 1
            print( '!!!Background Captured!!!')

        elif k == ord('r'):
            # press 'r' to reset the background
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0
            print ('!!!Reset BackGround!!!')

    cap.release()
    cv2.destroyAllWindows()
