
import urllib
import json 
from collections import OrderedDict
import pprint
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import cv2
import numpy
import os


data_dir_path = "D:/OCRtest/Clear/before"
file_list = os.listdir(data_dir_path)

imgs = []
imgs_after = []
# 全ファイル処理する
for file_name in file_list:
    root, ext = os.path.splitext(file_name)
    if ext == '.png' or '.jpeg' or '.jpg':

        # ファイルパス取得
        abs_name = data_dir_path + '/' + file_name
        
        # 画像読込
        img = cv2.imread(abs_name)
      
        # 白黒画像で読込 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        #ノイズ除去
        th1 = img.copy()
        t = 225
        #th1[img < t] = 0
        th1[img >= t] = 255
        cv2.imwrite('D:/OCRtest/Clear/after/' + file_name, th1)
        
        ##線をはっきりさせてみる
        #th1 = img.copy()
        #t = 220
        ## 方法1（NumPyで実装）
        #th1[img < t] = 0
        #th1[img >= t] = 255
        #th2 = cv2.bitwise_not(th1)
        #th3 = cv2.bitwise_not(th2)
        #cv2.imwrite('D:/OCRtest/Clear/after/' + file_name, th3)
        ##cv2.imwrite('T:/IT/Hackfest/Merge/clear/0/' + file_name, th1)
idx=0
for index, (image,result) in enumerate(imgs[:100]):
    plt.subplot(10, 10, index + 1)
    plt.axis('off')
    plt.title(result)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    idx=index    

idx = idx + 1

plt.show()

