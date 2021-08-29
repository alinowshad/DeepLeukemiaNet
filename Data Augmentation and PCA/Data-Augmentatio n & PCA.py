#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import essential libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import itertools
import keras
import seaborn as sns
from glob import glob
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import os
import cv2
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import keras.backend as K
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras import regularizers
from keras.callbacks import Callback
from keras import backend
from keras.models import load_model


# In[2]:


blood_cell_directory = os.path.join('..', 'input/lymphoblastsedit/')

# creating a directory for all images present with us and bringing them under same directory
image_directory = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(blood_cell_directory, '*', '*.jpg'))}


# In[3]:


NotHealthyPaths = []
for dirname, _, filenames in os.walk(os.path.join(blood_cell_directory, 'Kernels 4/guass One')):
    for filename in filenames:
        if (filename[-3:] == 'jpg'):
            NotHealthyPaths.append(os.path.join(dirname, filename))
NotHealthyPaths


# In[4]:


HealthyPaths = []
for dirname, _, filenames in os.walk(os.path.join(blood_cell_directory, 'Kernels 4/guass Two')):
    for filename in filenames:
        if (filename[-3:] == 'jpg'):
            HealthyPaths.append(os.path.join(dirname, filename))
HealthyPaths


# In[5]:


df = pd.DataFrame(columns = ['path', 'label'])


# In[6]:


# labels for Healthy:0, NotHealthy:1

for index1 in HealthyPaths:
    df=df.append({'path' : str(index1) , 'label' : 0} , ignore_index=True)
for index2 in NotHealthyPaths:
    df=df.append({'path' : str(index2) , 'label' : 1} , ignore_index=True)


# In[7]:


from skimage import color


# In[8]:


df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((224,224))))


# In[9]:


for index in range(len(df['image'])):
    df['image'][index] = color.rgb2gray(df['image'][index])


# In[10]:


df.head()


# In[11]:


# Printing Sample images for each lesion type
n_samples = 5
num_classes = 2
fig, m_axs = plt.subplots(num_classes, n_samples, figsize = (4*n_samples, 3*num_classes))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         df.sort_values(['label']).groupby('label')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)


# In[12]:


df['image'][0].shape


# In[13]:


# Importing necessary functions
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[14]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)


# In[15]:


features = np.asarray(df['image'].tolist())


# In[16]:


# Normalizing the data
data_mean = np.mean(features)
data_std = np.std(features)

Normilized_features = (features - data_mean)/data_std


# In[17]:


labels = df['label']


# In[18]:


Normilized_features.shape


# In[19]:


labels


# In[20]:


Normilized_features = Normilized_features.reshape((Normilized_features.shape[0], 224, 224, 1))
Normilized_features = Normilized_features.astype('float32')
# define data preparation


# In[21]:


Normilized_features.shape


# In[22]:


datagen.fit(Normilized_features)


# In[23]:


len(Normilized_features)


# In[24]:


Normilized_features[0].shape


# In[25]:


Images = []
Image_Labels = []
for i in range(len(Normilized_features)):
    Images.append(Normilized_features[i])
    Image_Labels.append(labels[i])


# In[26]:


from matplotlib import pyplot
for X_batch, y_batch in datagen.flow(Normilized_features, labels, batch_size=10):
    #Normilized_features = np.append(Normilized_features, X_batch[i])
    #labels = np.append(labels, y_batch[i])
    #print(len(X_batch))
    for j in range(len(X_batch)):
        X_batch[j] = np.asarray(X_batch[j])
        X_batch[j] = X_batch[j].astype('float32')
        Images.append(X_batch[j])
        Image_Labels.append(y_batch[j])
        #print(X_batch[j].shape)
        #print(y_batch[j])
    print("-------------------")
    print(len(Images))
    if len(Images) > 1200:
        break


# In[27]:


len(Image_Labels)


# In[28]:


dataframe = pd.DataFrame(columns = ['image', 'label'])


# In[29]:


for index in range(len(Image_Labels)):
    dataframe = dataframe.append({'image':Images[index]  , 'label': Image_Labels[index]} , ignore_index=True)


# In[30]:


dataframe.head()


# In[31]:


dataframe['label'].value_counts()


# In[32]:


dataframe['flat'] = dataframe['image']


# In[33]:


for index in range(len(dataframe['image'])):    
    dataframe['flat'][index] = dataframe['image'][index].reshape(-1,50176)


# In[34]:


dataframe.head()


# In[35]:


dataframe['flat'] = dataframe['flat'].tolist()


# In[36]:


for index in range(len(dataframe['flat'])):
    if isinstance(dataframe['flat'][index], np.ndarray):
        dataframe['flat'][index] = dataframe['flat'][index].tolist()


# In[37]:


df_features = pd.DataFrame(dataframe['flat'])


# In[38]:


df_features['label'] = dataframe['label']


# In[39]:


data_array = np.asarray(dataframe['image'].tolist())


# In[40]:


data_array = data_array.reshape(-1,50176)


# In[41]:


df_features = pd.DataFrame(data_array)


# In[42]:


df_features['label'] = dataframe['label']
print('Size of the dataframe: {}'.format(df_features.shape))


# In[43]:


print('Size of the dataframe: {}'.format(df_features.shape))


# In[44]:


df_features.to_csv('Data.csv')


# In[45]:


from sklearn.decomposition import PCA
pca = PCA(1200)
lower_dimension_data = pca.fit_transform(df_features.iloc[:,:-1])
lower_dimension_data.shape


# In[46]:


sum(pca.explained_variance_ratio_)


# In[47]:


#Project lower dimension data onto original features
approximation = pca.inverse_transform(lower_dimension_data)
print(approximation.shape)
approximation = approximation.reshape(-1,224,224, 1)
X_norm = data_array
X_norm = X_norm.reshape(-1,224,224, 1)


# In[52]:


fig4, axarr = plt.subplots(3,2,figsize=(8,8))
axarr[0,0].imshow(X_norm[0,],cmap='gray')
axarr[0,0].set_title('Original Image')
axarr[0,0].axis('off')
axarr[0,1].imshow(approximation[0,],cmap='gray')
axarr[0,1].set_title('99% Variance ')
axarr[0,1].axis('off')
axarr[1,0].imshow(X_norm[1,],cmap='gray')
axarr[1,0].set_title('Original Image')
axarr[1,0].axis('off')
axarr[1,1].imshow(approximation[1,],cmap='gray')
axarr[1,1].set_title('99% Variance ')
axarr[1,1].axis('off')
axarr[2,0].imshow(X_norm[2,],cmap='gray')
axarr[2,0].set_title('Original Image')
axarr[2,0].axis('off')
axarr[2,1].imshow(approximation[2,],cmap='gray')
axarr[2,1].set_title('99% Variance ')
axarr[2,1].axis('off')
plt.show()


# In[53]:


lower_dimension_data = pd.DataFrame(lower_dimension_data)


# In[54]:


lower_dimension_data['label'] = dataframe['label']
print('Size of the dataframe: {}'.format(lower_dimension_data.shape))


# In[55]:


lower_dimension_data.to_csv('PCA_Data.csv')


# In[ ]:




