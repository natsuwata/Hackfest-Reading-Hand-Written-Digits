
# coding: utf-8

# In[111]:


import cv2
import matplotlib.pyplot as plt
import numpy as np 
import http.client, urllib.request, urllib.parse, urllib.error
import json 

#Set the matplotlib figure default size
from IPython.core.pylabtools import figsize
import os, uuid, sys
from azure.storage.blob import BlockBlobService, PublicAccess


# In[112]:


figsize(48,24)
Line = 0.8   # 赤文字表記する基準


# In[113]:


data_dir_path = os.environ['DIR_NOW']
file_name = os.environ['FILE_NOW']
img_name = data_dir_path + '/' + file_name
img = cv2.imread(img_name)
plt.imshow(img)
plt.show()


# In[114]:


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()


# # 単純なしきい値処理
# 
# 右のリンクよりご確認してお願いいたします：[画像のしきい値処理](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html)
# 
# 方法は単純です．あるがその画素値がしきい値より大きければある値(白)を割り当て，そうでなければ別の値(黒)を割り当てます．関数は cv2.threshold を使います．第1引数は入力画像で グレースケール画像でなければいけません ．第2引数はしきい値で，画素値を識別するために使われます．第3引数は最大値でしきい値以上(指定するフラグ次第では以下)の値を持つ画素に対して割り当てられる値です．OpenCVは幾つかのしきい値処理を用意しており，第4引数にて指定します．以下がフラグの一覧です:
# 
# cv2.THRESH_BINARY
# cv2.THRESH_BINARY_INV
# cv2.THRESH_TRUNC
# cv2.THRESH_TOZERO
# cv2.THRESH_TOZERO_INV

# In[115]:


ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
plt.imshow(thresh)
plt.show()


# # モルフォロジー変換
# 
# 右のリンクよりご確認してお願いいたします：[モルフォロジー変換](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)
# 
# モルフォロジー変換は主に二値画像を対象とし，画像上に写っている図形に対して作用するシンプルな処理を指します.モルフォロジー変換には入力画像と 処理の性質を決める 構造的要素 ( カーネル )の二つを入力とします．基本的なモルフォロジー処理として，収縮(Erosion)と膨張(Dilation)が挙げられます．他には，この二つの処理を組み合わせたオープニングとクロージングといった処理も挙げられます．
# 
# ## 膨張(Dilation)
# 
# 収縮の逆の処理です．カーネル内に画素値が ‘1’ の画素が一つでも含まれれば，出力画像の注目画素の画素値を ‘1’ にします．画像中の白色の領域を増やすと言えますし，前景物体のサイズを増やすとも考えられます．普通は収縮の後に膨張させるノイズの除去方法で使われます．前景物体を膨張させるわけです．一度ノイズを消してしまえば，ノイズが再び発生することは無くなりますし，物体の領域も増えます．物体から離れてしまった部分を再び付けたい時に便利です．It is also useful in joining broken parts of an object.
# 

# In[116]:


dilate = cv2.dilate(thresh,None,iterations=2)
plt.imshow(dilate)
plt.show()


# # 輪郭の近似方法
# 
# 右のリンクよりご確認してお願いいたします：[輪郭とは何か？](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html)
# 
# 輪郭とは同じ色や値を持つ(境界線に沿った)連続する点をつなげて形成される曲線のことです．形状解析や物体追跡，認識お有用なツールになります．
# 
# ...近似方法のフラグを cv2.CHAIN_APPROX_NONE と指定すると輪郭上の全点の情報を保持します．しかし，本当に全点の情報を保持する必要があるでしょうか？例えば，直線の輪郭を検出したとして，検出した線を表現するのに境界上の全ての点の情報を保持する必要があるでしょうか？いいえ，直線の端点のみを保持するだけで十分でしょう． cv2.CHAIN_APPROX_SIMPLE フラグを指定すると，輪郭を圧縮して冗長な点の情報を削除し，メモリの使用を抑えられます．...
# 

# In[117]:


contours = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
print(contours)


# In[118]:


orig = img.copy()
for cnt in contours:
    
    if  cv2.contourArea(cnt)<1500000: #or cv2.contourArea(cnt)>1500000: 
        continue
    
    print(cv2.contourArea(cnt))
    
    x, y, w, h = cv2.boundingRect(cnt)   

    cv2.drawContours(orig,[cnt],0,(0,255,0),3)
    
    rRect = cv2.minAreaRect(cnt)
    
    angle = rRect[2] + 90
#     if rRect[1][0] < rRect[1][1]:
#       angle = angle - 90;
    
    if(angle>45):
        angle = angle-90
    
    print(angle)
    num_rows, num_cols = orig.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
    orig = cv2.warpAffine(orig, rotation_matrix, (num_cols, num_rows))
    
#     # 番号
#     cv2.rectangle(orig,(x+int(0.596*w),y+int(0.172*h)),(x+int(0.997*w),y+int(0.301*h)),(0,0,255),2)
#     #記号１
#     cv2.rectangle(orig,(x+int(0.213*w),y+int(0.172*h)),(x+int(0.476*w),y+int(0.300*h)),(0,0,255),2)
#     #記号２
#     cv2.rectangle(orig,(x+int(0.526*w),y+int(0.21*h)),(x+int(0.579*w),y+int(0.301*h)),(0,0,255),2)
    

plt.imshow(orig)
plt.show()


# In[119]:


for cnt in contours:
    
    if  cv2.contourArea(cnt)<50000 or cv2.contourArea(cnt)>55000: 
        continue
        
    print(cv2.contourArea(cnt))
    
    x1, y1, w1, h1 = cv2.boundingRect(cnt)   
#     cv2.rectangle(orig,(x1,y1+h1),(x1+w1,y1+int(2.4*h1)),(0,0,255),2)


# In[120]:


# orig = img.copy()
# for cnt in contours:
#     if  cv2.contourArea(cnt)<1500000: 
#         continue
#     x, y, w, h = cv2.boundingRect(cnt)   
#     cv2.rectangle(orig,(x,y+h),(x+w,y+int(h*2.4)),(0,255,0),2)
# plt.imshow(orig)
# plt.show()
print( x, y, w, h)


# In[121]:


# 社員コードの取得

num_digi = int(w1/8)
print(num_digi)

for j in range(0,8):
   # crop_image = orig[(y-int(0.66*h))+10:(y-int(0.564*h)-3),x+int(0.615*w)+j*num_digi+10:(x+int(0.615*w)+(j+1)*num_digi-10)]
    crop_image = orig[(y1+h1)+10:(y1+int(2.4*h1)-10),x1+j*num_digi+10:(x1+(j+1)*num_digi-10)]
    crop_h, crop_w, _ = crop_image.shape 
    print(crop_h)
    
    gray_k = cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
    _,thresh_k = cv2.threshold(gray_k,127,255,cv2.THRESH_BINARY_INV)
    dilate_k = cv2.dilate(thresh_k,None,iterations=2)

    cnts = cv2.findContours(dilate_k,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    if len(cnts) > 0:
        white_img = np.zeros((crop_h, crop_h, 3), dtype=np.uint8)
        white_img.fill(255)
        white_img[0:crop_h, int((crop_h-crop_w)/2):int((crop_h-crop_w)/2)+crop_w] = crop_image
        
        img2 = cv2.resize(white_img, (28, 28), 0, 0, cv2.INTER_NEAREST)
        cv2.imwrite("./Output/StaffCode_" + str(j) + ".jpg", img2)
        
        headers = {
                # Request headers
                'Content-Type': 'application/octet-stream',
                'Prediction-Key': '****',
            }
        
        try:
            conn = http.client.HTTPSConnection('southcentralus.api.cognitive.microsoft.com')
            conn.request("POST", "****",  
                open("./Output/StaffCode_" + str(j) + ".jpg", 'rb', buffering=0).readall(), headers)
            response = conn.getresponse()
            data = response.read()
            
            r = data.decode()
            js = json.loads(r)
            data_tagName = js['predictions'][0]['tagName']
            data_probability = js['predictions'][0]['probability']
            strPlot = ("値：" + data_tagName + "    確率：" + str(data_probability))
            print(strPlot)
            # 確率によって色を設定する
            if data_probability < Line:
                col = (255, 0, 0)
            else:
                col = (0, 0, 255) 

            cv2.putText(orig, data_tagName, ( (x+int(0.615*w)+j*num_digi+10),(y-int(0.66*h)+30) ), cv2.FONT_HERSHEY_PLAIN, 4, col, 5, cv2.LINE_AA)
            cv2.putText(orig,  str(data_probability)[0:4], ( (x+int(0.615*w)+j*num_digi+10),(y-int(0.564*h)+40) ), cv2.FONT_HERSHEY_PLAIN, 2, col, 2, cv2.LINE_AA)

            print("./Output/StaffCode_" + str(j) + ".jpg")
            #print(data)
            conn.close()
        except Exception as e:
            print(e)
plt.imshow(orig)
plt.show()


# In[122]:


# 記号の取得

num_digi = int(w*0.262/5)
print(num_digi)

for j in range(0,5):
    crop_image = orig[(y+int(0.171*h))+10:(y+int(0.3*h)-10),x+int(0.213*w)+j*num_digi+10:(x+int(0.213*w)+(j+1)*num_digi-10)]
    crop_h, crop_w, _ = crop_image.shape 
    
    gray_k = cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
    _,thresh_k = cv2.threshold(gray_k,127,255,cv2.THRESH_BINARY_INV)
    dilate_k = cv2.dilate(thresh_k,None,iterations=2)

    cnts = cv2.findContours(dilate_k,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    if len(cnts) > 0:
        white_img = np.zeros((crop_h, crop_h, 3), dtype=np.uint8)
        white_img.fill(255)
        white_img[0:crop_h, int((crop_h-crop_w)/2):int((crop_h-crop_w)/2)+crop_w] = crop_image
        
        img2 = cv2.resize(white_img, (28, 28), 0, 0, cv2.INTER_NEAREST)
        cv2.imwrite("./Output/BankCodeK_" + str(j) + ".jpg", img2)
        
        headers = {
                # Request headers
                'Content-Type': 'application/octet-stream',
                'Prediction-Key': '****',
            }
        
        try:
            conn = http.client.HTTPSConnection('southcentralus.api.cognitive.microsoft.com')
            conn.request("POST", "****",  
                open("./Output/StaffCode_" + str(j) + ".jpg", 'rb', buffering=0).readall(), headers)
            response = conn.getresponse()
            data = response.read()
            
            r = data.decode()
            js = json.loads(r)
            data_tagName = js['predictions'][0]['tagName']
            data_probability = js['predictions'][0]['probability']
            strPlot = ("値：" + data_tagName + "    確率：" + str(data_probability))
            print(strPlot)
            # 確率によって色を設定する
            if data_probability < Line:
                col = (255, 0, 0)
            else:
                col = (0, 0, 255) 
                
            cv2.putText(orig, data_tagName, ( (x+int(0.213*w)+j*num_digi+10),(y+int(0.171*h)+10) ), cv2.FONT_HERSHEY_PLAIN, 4, col, 5, cv2.LINE_AA)
            cv2.putText(orig,  str(data_probability)[0:4], ( (x+int(0.213*w)+j*num_digi+10),(y+int(0.3*h)+30) ), cv2.FONT_HERSHEY_PLAIN, 2, col, 2, cv2.LINE_AA)

            print("./Output/BankCodeK_" + str(j) + ".jpg")
            print(data)
            conn.close()
        except Exception as e:
            print(e)
plt.imshow(orig)
plt.show()


# In[123]:


# 記号(※)の取得

crop_image = orig[(y+int(0.21*h)+10):(y+int(0.301*h)-10),x+int(0.526*w)+10:(x+int(0.579*w)-10)]
crop_h, crop_w, _ = crop_image.shape 

gray_k = cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
_,thresh_k = cv2.threshold(gray_k,127,255,cv2.THRESH_BINARY_INV)
dilate_k = cv2.dilate(thresh_k,None,iterations=2)

cnts = cv2.findContours(dilate_k,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]

if len(cnts) > 0:

    white_img = np.zeros((crop_h, crop_h, 3), dtype=np.uint8)
    white_img.fill(255)
    white_img[0:crop_h, int((crop_h-crop_w)/2):int((crop_h-crop_w)/2)+crop_w] = crop_image

    img2 = cv2.resize(white_img, (28, 28), 0, 0, cv2.INTER_NEAREST)
    cv2.imwrite("./Output/BankCodeK_6" + ".jpg", img2)
    
    
    headers = {
            # Request headers
            'Content-Type': 'application/octet-stream',
            'Prediction-Key': '****',
        }

    try:
        conn = http.client.HTTPSConnection('southcentralus.api.cognitive.microsoft.com')
        conn.request("POST", "****",  
            open("./Output/BankCodeK_6" + ".jpg", 'rb', buffering=0).readall(), headers)
        response = conn.getresponse()
        data = response.read()
                  
        r = data.decode()
        js = json.loads(r)
        data_tagName = js['predictions'][0]['tagName']
        data_probability = js['predictions'][0]['probability']
        strPlot = ("値：" + data_tagName + "    確率：" + str(data_probability))
        print(strPlot)
        # 確率によって色を設定する
        if data_probability < Line:
            col = (255, 0, 0)
        else:
            col = (0, 0, 255) 
           
        cv2.putText(orig, data_tagName, ( (x+int(0.526*w)),(y+int(0.21*h)+10) ), cv2.FONT_HERSHEY_PLAIN, 4, col, 5, cv2.LINE_AA)
        cv2.putText(orig,  str(data_probability)[0:4], ( (x+int(0.526*w)),(y+int(0.301*h)+30) ), cv2.FONT_HERSHEY_PLAIN, 2, col, 2, cv2.LINE_AA)

            
        print(data)
        conn.close()
    except Exception as e:
        print(e)

plt.imshow(orig)
plt.show()


# In[124]:


# 番号の取得

num_digi = int(w*0.406/8)
print(num_digi)

for k in range(0,8):
    crop_image = orig[(y+int(0.172*h))+10:(y+int(0.297*h)-5),x+int(0.596*w)+k*num_digi+10:(x+int(0.596*w)+(k+1)*num_digi-10)]
    crop_h, crop_w, _ = crop_image.shape 
    
#     cv2.rectangle(orig,(x+int(0.596*w)+k*num_digi+5,(y+int(0.172*h))+5),((x+int(0.596*w)+(k+1)*num_digi-5),(y+int(0.297*h)-5)),(0,0,255),2)

    gray_k = cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
    _,thresh_k = cv2.threshold(gray_k,127,255,cv2.THRESH_BINARY_INV)
    dilate_k = cv2.dilate(thresh_k,None,iterations=2)

    cnts = cv2.findContours(dilate_k,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    if len(cnts) > 0:    

        white_img = np.zeros((crop_h, crop_h, 3), dtype=np.uint8)
        white_img.fill(255)
        white_img[0:crop_h, int((crop_h-crop_w)/2):int((crop_h-crop_w)/2)+crop_w] = crop_image

    #     plt.imshow(white_img)
    #     plt.show()

        img2 = cv2.resize(white_img, (28, 28), 0, 0, cv2.INTER_NEAREST)
        cv2.imwrite("./Output/BankCodeB_" + str(k) + ".jpg", img2)

        headers = {
                # Request headers
                'Content-Type': 'application/octet-stream',
                'Prediction-Key': '****',
            }
        
        try:
            conn = http.client.HTTPSConnection('southcentralus.api.cognitive.microsoft.com')
            conn.request("POST", "****",  
                open("./Output/StaffCode_" + str(j) + ".jpg", 'rb', buffering=0).readall(), headers)
            response = conn.getresponse()
            data = response.read()

            r = data.decode()
            js = json.loads(r)
            data_tagName = js['predictions'][0]['tagName']
            data_probability = js['predictions'][0]['probability']
            strPlot = ("値：" + data_tagName + "    確率：" + str(data_probability))
            print(strPlot)
            # 確率によって色を設定する
            if data_probability < Line:
                col = (255, 0, 0)
            else:
                col = (0, 0, 255) 

            cv2.putText(orig, data_tagName, ( (x+int(0.596*w)+k*num_digi+10),((y+int(0.172*h))+10) ), cv2.FONT_HERSHEY_PLAIN, 4, col, 5, cv2.LINE_AA)
            cv2.putText(orig,  str(data_probability)[0:4], ( (x+int(0.596*w)+k*num_digi+10),((y+int(0.297*h)+40)) ), cv2.FONT_HERSHEY_PLAIN, 2, col, 2, cv2.LINE_AA)

            print("BankCodeB_" + str(k) + ".jpg")

            print(data)
            conn.close()
        except Exception as e:
            print(e)


# In[125]:


plt.imshow(orig)
plt.show()


# In[126]:


from azure.storage.blob import BlockBlobService
try:
    account_name='****'
    account_key= '****'
    container_name='handwrittenblobs'
    #blob_name='test'
    local_path=os.getcwd()+ "/Result/"
    #local_file_name = file_name
    print(local_path  + file_name)
    
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    cv2.imwrite(local_path + file_name, orig_rgb)
    
    full_path_to_file =os.path.join(local_path, file_name)
    
    service = block_blob_service = BlockBlobService(account_name=account_name, account_key=account_key)
    blob = service.create_blob_from_path(container_name,file_name,full_path_to_file)

except Exception as e:
    print(e)

