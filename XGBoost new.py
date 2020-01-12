#!/usr/bin/env python
# coding: utf-8

# In[1]:


#all the neceessary libraries are imported
import numpy as np
import matplotlib.pyplot as plt
import random 

from pathlib import Path

from sklearn import metrics
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

from xgboost import XGBClassifier

from skimage.io import imread,imshow
from skimage.transform import resize

from mlxtend.plotting import plot_confusion_matrix

from datetime import datetime


# In[2]:


##images are loaded with this function by recursively reading all the folders

def read_image_files(folder_path, size=(128,128)):

    image_directory = Path(folder_path)
    folders = [directory for directory in image_directory.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    images = []
    target = []
    flattened_data = []
  
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            img_resized = resize(img, size, anti_aliasing=True, mode='reflect')
            
            images.append(img_resized)
            target.append(i)
            flattened_data.append(img_resized.flatten()) 
            
    images = np.array(images)
    target = np.array(target)
    flattened_data = np.array(flattened_data)
    
    return Bunch(data=flattened_data,
                 target=target,
                 target_names=categories,
                 images=images
                 )


# In[3]:


tomato_dataset = read_image_files("C://NCI/Sem 3/tomato plant disease dataset/PlantVillage/")


# In[4]:


print(len(tomato_dataset.images))


# In[5]:


imshow(tomato_dataset.images[800])


# In[6]:


tomato_dataset.target[6500]


# In[7]:


tomato_dataset.target_names[tomato_dataset.target[6500]]


# In[8]:


value = []
Tomato_healthy = Tomato_Late_blight = Tomato_Leaf_Mold = Tomato_Septoria_leaf_spot = Tomato_mosaic_virus = 0

for i in range(len(tomato_dataset.images)):
    if(tomato_dataset.target[i] == 0):
        Tomato_healthy = Tomato_healthy + 1
    elif (tomato_dataset.target[i] == 1):
         Tomato_Late_blight = Tomato_Late_blight + 1
    elif (tomato_dataset.target[i] == 2):
        Tomato_Leaf_Mold = Tomato_Leaf_Mold + 1
    elif (tomato_dataset.target[i] == 3):
        Tomato_mosaic_virus = Tomato_mosaic_virus + 1
    else:
        Tomato_Septoria_leaf_spot = Tomato_Septoria_leaf_spot + 1
        
value=[Tomato_mosaic_virus,Tomato_healthy,Tomato_Late_blight,Tomato_Leaf_Mold,Tomato_Septoria_leaf_spot]
types = ('mosaic_virus','Healthy','Late_blight','Leaf_Mold','Septoria_leaf_spot')
y_pos = np.arange(5)

plt.figure(figsize=(10,8))
plt.bar(y_pos, value, align='center', alpha=0.5)
plt.xticks(y_pos, types)
plt.ylabel('Number of Images')
plt.title('Data Distribution')

plt.show()


# In[9]:


count = list(range(len(tomato_dataset.images)))
random_indexes = random.sample(count, 4)


# In[10]:


j = 1
plt.figure( figsize=(10, 8))
for i in random_indexes:
    plt.subplot(2, 2, j);
    plt.grid(False)
    plt.imshow(tomato_dataset.images[i])
    plt.title(tomato_dataset.target_names[tomato_dataset.target[i]])
    j = j + 1
plt.show()


# In[11]:


#test and train split
x_train, x_test, y_train, y_test = train_test_split(
    tomato_dataset.data, tomato_dataset.target, test_size=0.2,random_state=2002)


# In[12]:


y_test.shape


# In[13]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[14]:


start_time = timer(None) 


# In[1]:


#training the model
classifier = XGBClassifier(probability=True)
classifier.fit(x_train, y_train)


# In[16]:


timer(start_time)


# In[17]:


#evaluate the model
y_pred = classifier.predict(x_test)


# In[23]:


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[24]:


fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix,
                                show_absolute=True,
                                show_normed=False,
                                colorbar=True,
                               )
plt.show()


# In[18]:


print("Classification report for - \n{}:\n{}\n".format(
    classifier, metrics.classification_report(y_test, y_pred , target_names = types)))


# In[20]:


probabilities = classifier.predict_proba(x_test)

# select the probabilities for label 1.0
y_proba = probabilities[:,1]

# calculate false positive rate and true positive rate at different thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)
precision, recall, thresholds = precision_recall_curve(y_test, y_proba, pos_label=1)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label='AUC = {:0.2f}'.format(roc_auc))

plt.legend(loc=0, fontsize=10)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=16);


# In[21]:


plt.plot(precision, recall, label='Precision-recall curve')
_ = plt.xlabel('Precision',fontsize=16)
_ = plt.ylabel('Recall',fontsize=16)
_ = plt.title('Precision-recall curve',fontsize=18)
_ = plt.legend(loc="lower left",fontsize=10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




