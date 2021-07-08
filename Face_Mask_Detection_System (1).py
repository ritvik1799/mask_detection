#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# Mount to google drive
import os
# Set directory path for dataset
os.chdir("/content/drive/MyDrive")
dataset ='Dataset_Mask'
data_dir = os.listdir(dataset)
print(data_dir)


# In[ ]:


# import necessary library
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

img_row,img_col = 112,112

images =[]
labels = []

for category in data_dir:
  folder_path = os.path.join(dataset,category)
  for img in os.listdir(folder_path):
    img_path = os.path.join(folder_path,img)
    img = cv2.imread(img_path)

    try:
      # converting the image into grescale
      grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

      # resize the image to 56*56 to keep the size consistent
      resized_img = cv2.resize(grey_img,(img_row,img_col))
      images.append(resized_img)
      labels.append(category)
    # exception handling in case any error occur
    except Exception as e:
      print('Exception',e)

images = np.array(images)/255.0
print(images.shape)
images = np.reshape(images,(images.shape[0],img_row,img_col,1))

# Perform onehot encoding on the label since the label are in textual form
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
labels = np.array(labels)


# In[ ]:


print(images.shape)
x_train,x_test,y_train,y_test = train_test_split(images,labels,test_size = 0.25,random_state=0)


# In[ ]:





# **Buliding CNN model and create the architecture**

# In[ ]:


# import necessary libraries
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D

# define model parameter
num_classes = 2
batch_size = 32

# bulid cnn model using sequential api
model = Sequential()

# add first layer containg convolution,maxpool,activation
model.add(Conv2D(64,(3,3),input_shape = (img_row,img_col,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# add second layer containg convolution,maxpool,activation
model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# Flatten and dropout layer to stack the output convolution above as well as cater overfitting
model.add(Flatten())
model.add(Dropout(0.5))

# softmax classifier
model.add(Dense(64,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

print(model.summary())


# **Plot the model**
# 

# In[ ]:


from keras.utils.vis_utils import plot_model
plot_model(model,to_file = 'face_mask_detection_architecture.png')


# **Train the Model**

# In[ ]:


from keras.optimizers import Adam

epoch= 60
model.compile(loss = 'categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

fitted_model = model.fit(x_train,y_train,epochs=epoch,batch_size=batch_size,validation_split = 0.25 )


# In[ ]:


from matplotlib import pyplot as plt
# ploting train and validation loss
plt.plot(fitted_model.history['loss'],'r',label = 'training_loss')
plt.plot(fitted_model.history['val_loss'],'b',label = 'validation_loss')
plt.xlabel('Number of epochs')
plt.ylabel('accuray value')
plt.legend()
plt.show()

# ploting train and validation accuracy
plt.plot(fitted_model.history['accuracy'],'r',label = 'training_accuracy')
plt.plot(fitted_model.history['val_accuracy'],'b',label = 'validation_accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('accuray value')
plt.legend()
plt.show()


# In[ ]:


# save the model
model.save("face_mask_detection_system_1.h5")


# In[26]:


from tensorflow.keras.models import load_model
import cv2
import numpy as np


# In[30]:


model = load_model("face_mask_detection_system_1.h5")
face_det_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# In[ ]:


vid_source = cv2.VideoCapture(0)
text_dict = {0:'Mask ON',1:'NO Mask'}
rect_color_dict = {0:(0,255,0),1:(0,0,255)}

while(True):
    ret,img = vid_source.read()
    grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_det_classifier.detectMultiScale(grayscale_img,1.3,5)
    
    for (x,y,w,h) in faces:
        face_img = grayscale_img[y:y+h,x:x+w]
        resized_img = cv2.resize(face_img,(112,112))
        normalized_img = resized_img/255.0
        reshaped_img = np.reshape(normalized_img,(1,112,112,1))
        result = model.predict(reshaped_img)
        
        label = np.argmax(result,axix = 1)[0]
        
        cv2.rectangle(img,(x,y),(x+w,y+h),rect_color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),rect_color_dict[label],-1)
        cv2.putText(img,text_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        
    cv2.imshow('live feed',img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
vid_source.release()
cv2.destroyAllWindows()


# In[ ]:




