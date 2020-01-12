#!/usr/bin/env python
# coding: utf-8

# In[1]:


#all the neceessary libraries are imported 
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from os import listdir
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc,precision_recall_curve,confusion_matrix
from sklearn import metrics

from skimage.io import imread,imshow

from mlxtend.plotting import plot_confusion_matrix
from datetime import datetime


# In[2]:


#Fuction to calculate the time difference
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[3]:


#Epochs, Learning rates, image size and batch size are defined here
EPOCHS = 25
INIT_LR = 0.01
BS = 32
default_image_size = tuple((128, 128))
image_size = 0
directory_root = 'C://NCI/Sem 3/tomato plant disease dataset/PlantVillage/'
width=128
height=128
depth=3


# In[4]:


#converting images to array
images=[]
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)
            images.append(image)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


# In[5]:


#images are loaded with this function by recursively reading all the folders
image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    plant_disease_folder_list = listdir(directory_root)
    print (plant_disease_folder_list)

    for plant_disease_folder in plant_disease_folder_list:
        print(f"[INFO] Processing {plant_disease_folder} ...")
        plant_disease_image_list = listdir(f"{directory_root}/{plant_disease_folder}/")
            

        for image in plant_disease_image_list[:]:
            image_directory = f"{directory_root}/{plant_disease_folder}/{image}"
            if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                image_list.append(convert_image_to_array(image_directory))
                label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")


# In[6]:


image_size = len(image_list)
print(image_size)


# In[7]:


imshow(images[600])


# In[8]:


#encodes the label data
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
n_classes = len(label_binarizer.classes_)


# In[9]:


print(label_binarizer.classes_)


# In[10]:


value = []
Tomato_mosaic_virus = Tomato_healthy = Tomato_Late_blight = Tomato_Leaf_Mold = Tomato_Septoria_leaf_spot = 0

for i in range(len(label_list)):
    if (label_list[i] == 'Tomato__Tomato_mosaic_virus'):
        Tomato_mosaic_virus = Tomato_mosaic_virus + 1
    elif (label_list[i] == 'Tomato_healthy'):
        Tomato_healthy = Tomato_healthy + 1
    elif (label_list[i] == 'Tomato_Late_blight'):
        Tomato_Late_blight = Tomato_Late_blight + 1
    elif (label_list[i] == 'Tomato_Leaf_Mold'):
        Tomato_Leaf_Mold = Tomato_Leaf_Mold + 1    
    else:
        TTomato_Septoria_leaf_spot = Tomato_Septoria_leaf_spot + 1
        
value=[Tomato_mosaic_virus,Tomato_healthy,Tomato_Late_blight,Tomato_Leaf_Mold,Tomato_Septoria_leaf_spot]
types = ('mosaic_virus','Healthy','Late_blight','Leaf_Mold','Septoria_leaf_spot','Two_spotted_spider_mite')
y_pos = np.arange(5)

plt.figure(figsize=(10,10))
plt.bar(y_pos, value, align='center', alpha=0.5)
plt.xticks(y_pos, types)
plt.ylabel('Number of Images')
plt.title('Data Distribution')

plt.show()


# In[11]:


np_image_list = np.array(image_list, dtype=np.float16) / 255.0


# In[12]:


all_indexes = list(range(len(images)))
random_indexes = random.sample( all_indexes, 4)


# In[13]:


j = 1
plt.figure( figsize=(15, 6))
for i in random_indexes:
    plt.subplot(2, 2, j);
    plt.grid(False)
    plt.imshow(images[i])
   # plt.title(tomato_dataset.target_names[tomato_dataset.target[i]])
    j = j + 1
plt.show()


# In[14]:


#test and train split
print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 


# In[15]:


y_test.shape


# In[16]:


#data augmentation
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")


# In[22]:


# 5 layered CNN model is built here
model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
    
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="same",input_shape=inputShape, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding="same",input_shape=inputShape, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding="same",input_shape=inputShape, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(GlobalAveragePooling2D())
model.add(Dense(units=256, activation = "relu"))
model.add(Dropout(0.25))

model.add(Dense(units=128, activation = "relu"))
model.add(Dropout(0.25))

model.add(Dense(units=n_classes, activation = "softmax"))


# In[23]:


model.summary()


# In[24]:


model.compile(loss= "binary_crossentropy",optimizer = 'Adam',metrics=["accuracy"])


# In[25]:


start_time = timer(None)


# In[26]:


#training
history= model.fit_generator(
    aug.flow(x_train,y_train,batch_size=BS),
    validation_data=(x_test,y_test),
    steps_per_epoch =len(x_train) // BS,
    epochs=EPOCHS, verbose=1
)


# In[27]:


timer(start_time)


# In[28]:


#model is evaluated here
Y_pred= model.predict(x_test)


# In[29]:


y_pred = np.argmax(Y_pred,axis=1)


# In[30]:


y_test_end = label_binarizer.inverse_transform(y_test)


# In[31]:


label = LabelEncoder()
y_encoded = label.fit_transform(y_test_end)


# In[ ]:


def plot_accuracy(hist):
    plt.plot(hist['accuracy'])
    plt.plot(hist['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 
                'test'], 
               loc='upper left')
    plt.show()


# In[ ]:


plot_accuracy(history.history)


# In[ ]:


def plot_loss(hist):
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 
                'test'], 
               loc='upper left')
    plt.show()


# In[ ]:


plot_loss(history.history)


# In[32]:


cm = confusion_matrix(y_encoded, y_pred)
print(cm)


# In[33]:


fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=False,
                                colorbar=True,
                               )
plt.show()


# In[34]:


print("Classification report for - \n{}:\n{}\n".format(
    model, metrics.classification_report(y_encoded, y_pred)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




