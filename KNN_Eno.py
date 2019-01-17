
# coding: utf-8

# # Define function to return an array of file names

# In[240]:


import os

def return_list_of_files(rootdir, printname=False):
    all_files = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            all_files.append(os.path.join(subdir, file))
            if printname: 
                print(os.path.join(subdir, file))
    return np.asarray(all_files)    


# # Function to load data from file names into features + labels

# In[241]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def load_data(dataset_path):
    images_list = return_list_of_files(dataset_path)
    
    #print(images_list)
    print(len(images_list))
    
    features = np.ndarray(shape=(len(images_list), 28, 28),
                    dtype=np.uint8)
    labels = []
    for i in range(len(images_list)):
        try:
            im = mpimg.imread(images_list[i])

            features[i] = im
            #features[i] = im.flatten()
            labels.append(images_list[i].split("/")[2])
        except:
            print(images_list[i])
        
    return features, np.asarray(labels)


# In[242]:


features, labels = load_data("../TrainingDataAll")
test_features, test_labels = load_data("../TestData")


# In[243]:


print("\n", features.shape, "\n", labels.shape)
print("\n", test_features.shape, "\n", test_labels.shape)


# In[244]:


plt.imshow(test_features[3].reshape(28, 28))


# # Label encoder to convert string labels into integers

# In[245]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(labels)
labels_encoded = le.transform(labels) 

test_le = preprocessing.LabelEncoder()
test_le.fit(test_labels)
test_labels_encoded = test_le.transform(test_labels) 

#list(le.inverse_transform([2, 2, 1]))


# In[246]:


le.classes_
test_le.classes_


# In[247]:


from sklearn.model_selection import train_test_split

X_train=features
X_test=test_features
Y_train=labels_encoded
Y_test=test_labels_encoded

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(X_train.shape))
print("Test data   : {}".format(X_test.shape))
print("Train labels: {}".format(Y_train.shape))
print("Test labels : {}".format(Y_test.shape))


# In[248]:


#28×28 の二次元で表現されている入力となる画像の情報が、784個になるようにしたいので、1次元になるように変形させる
X_train  = X_train.reshape(len(X_train), 784)
X_test   = X_test.reshape(len(X_test), 784)


# In[249]:


print(X_train.shape)
print(Y_train.shape)


# In[253]:


import numpy as np
import cv2
from matplotlib import pyplot as plt

# Now we prepare train_data and test_data.
train = X_train.astype(np.float32)
test = X_test.astype(np.float32)


# Create labels for train and test data
train_labels = Y_train[:,np.newaxis]
test_labels = Y_test[:,np.newaxis]

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create() 
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels) 
ret,result,neighbours,dist = knn.findNearest(test,k=3)
#print(result)
#print(test_labels)
#print(result.size)
# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
print(correct)
accuracy = correct/result.size*100
print(accuracy)

