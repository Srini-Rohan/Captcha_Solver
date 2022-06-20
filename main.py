import cv2
import numpy as np
from PIL import ImageFilter
import skimage.exposure
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, GlobalAveragePooling2D,InputLayer,Activation,MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

img=cv2.imread('name.jpeg',0)
img=cv2.bitwise_not(img)

img = cv2.resize(img, (700, 280),
               interpolation = cv2.INTER_NEAREST)

img=img[10:-10,10:-10]

ret,img = cv2.threshold(img,130,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img_letter=[]
img_emoji=[]
dic={}
lists=[]
for i in range(len(contours)):
    if(cv2.contourArea(contours[i])>200):
        (x,y,w,h)=cv2.boundingRect(contours[i])
        imgl=img[y-5:y+h+5,x-5:x+w+5]
        if(imgl.shape[0]==0 or imgl.shape[1]==0):
            continue
        imgf=np.zeros((28,28))
        imge1=np.zeros((300,300))
        imgl = cv2.resize(imgl, (24, 24),interpolation = cv2.INTER_NEAREST)
        imge=cv2.resize(imgl, (200, 200),interpolation = cv2.INTER_NEAREST)
        kernel = np.ones((2,2), np.uint8)
        imgl = cv2.dilate(imgl, kernel, iterations=1)
        imge = cv2.dilate(imge, kernel, iterations=1)
        imgf[2:26,2:26]=imgl
        imge1[50:250,50:250]=imgl=imge
        lists.append(x)
        img_letter.append(imgf)
        img_emoji.append(imge1)
model_letter=tf.keras.models.load_model('model_letters.h5')
model_emoji=tf.keras.models.load_model('model_emojis.h5')
mapping_letter="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
mapping_emoji="1324567"
for i in range(len(img_letter)):
    image=np.reshape(img_letter[i],(1,28,28,1))/255
    ans=model_letter.predict(image)
    if(ans[0][np.argmax(ans)]>0.98):
        dic[lists[i]]=mapping_letter[np.argmax(ans)]
    else:
        image=np.reshape(img_emoji[i],(1,300,300,1))/255
        ans=model_emoji.predict(image)
        dic[lists[i]]=mapping_emoji[np.argmax(ans)]

lists.sort()
for i in lists:
  print(dic[i],end="")
