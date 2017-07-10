
# coding: utf-8

# In[1]:

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Lambda, Flatten
from keras.optimizers import Adam


# In[2]:

log_file = 'data/data/driving_log.csv'
img_folder = 'data/data/'


# In[3]:

logs = []
with open(log_file, 'rt') as f:
    log_reader = csv.reader(f)
    for line in log_reader:
        logs.append(line)
labels = logs.pop(0)


# In[4]:

def image_preprocessing(img):
    # convert the image to HSV from RGB
    # extract the S channel
    # crop the image
    # resize the image
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_channel = img_hsv[:, :, 1]
    cropped = s_channel[53:140,: ]
    resized = cv2.resize(cropped, (32, 16))
    return resized


# In[5]:

data = dict()
data['features']= []
data['label'] = []


# In[6]:

# load center camera image
for i in range(len(logs)):
    # center image
    img_path = img_folder + logs[i][0]
    img = plt.imread(img_path)
    data['features'].append(image_preprocessing(img))
    data['label'].append(float(logs[i][3]))
    
    # left image with correction
    img_path = img_folder + (logs[i][1])[1:]
    img = plt.imread(img_path)
    data['features'].append(image_preprocessing(img))
    data['label'].append(float(logs[i][3]) + 0.3)
    
    # right image with correction
    img_path = img_folder + logs[i][2][1:]
    img = plt.imread(img_path)
    data['features'].append(image_preprocessing(img))
    data['label'].append(float(logs[i][3]) - 0.3)


# In[7]:

X_train = np.array(data['features']).astype('float32')
y_train = np.array(data['label']).astype('float32')

# flip the images to create more data
X_train = np.append(X_train,X_train[:,:,::-1],axis=0)
y_train = np.append(y_train,-y_train,axis=0)

print(len(X_train))
print(len(y_train))


# In[8]:

X_train, y_train = shuffle(X_train, y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.1)


# In[9]:
# since our new image is just one channel, it is important to specify that
X_train = X_train.reshape(X_train.shape[0], 16, 32, 1)
X_val = X_val.reshape(X_val.shape[0], 16, 32, 1)


# In[11]:

model = Sequential([
        Lambda(lambda x: x/127.5 - 1.,input_shape=(16,32,1)),
        Conv2D(16, 3, 3, border_mode='valid', activation='relu'),
        MaxPooling2D((2,2),(2,2),'valid'),
        Dropout(0.25),
        Conv2D(32, 3, 3, border_mode='valid', activation='relu'),
        MaxPooling2D((2,2),(2,2),'valid'),
        Dropout(0.25),

        Flatten(),
        Dense(1)
    ])

model.summary()


# In[12]:

model.compile(loss='mean_squared_error',optimizer='adam')
history = model.fit(X_train, y_train,batch_size=128, nb_epoch=10,verbose=1, validation_data=(X_val, y_val))


# In[13]:

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Model saved.")


# In[ ]:



