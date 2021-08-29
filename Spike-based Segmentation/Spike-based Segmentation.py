#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms
import struct
import glob


# In[14]:


import SpykeTorch.utils as utils

kernels = [ utils.GaborKernel(window_size = 4, orientation = 45+22.5),
            utils.GaborKernel(4, 90+22.5),
            utils.GaborKernel(4, 135+22.5),
            utils.GaborKernel(4, 180+22.5)]
filter = utils.Filter(kernels, use_abs = True)


# In[15]:


def time_dim(input):
    return input.unsqueeze(0)


# In[16]:


import matplotlib.pyplot as plt
import random

dataset = ImageFolder("dataset/eth")
sample_idx = random.randint(0, len(dataset) - 1)

# plotting the sample image
ax = plt.subplot(1,1,1)
plt.setp(ax, xticklabels=[])
plt.setp(ax, yticklabels=[])
plt.xticks([])
plt.yticks([])
plt.imshow(dataset[sample_idx][0])
plt.show()


# In[17]:


import SpykeTorch.utils as utils
import SpykeTorch.functional as sf
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.Grayscale(),
    transforms.ToTensor(),
    time_dim,
    filter,
    sf.pointwise_inhibition,
    utils.Intensity2Latency(number_of_spike_bins = 15, to_spike = True)])


# In[18]:


import numpy as np

dataset = ImageFolder("dataset/eth", transform) # adding transform to the dataset
plt.style.use('seaborn-white')
plt_idx = 0
sw = dataset[sample_idx][0]
for f in range(4):
    for t in range(5):
        plt_idx += 1
        ax = plt.subplot(5, 5, plt_idx)
        plt.setp(ax, xticklabels=[])
        plt.setp(ax, yticklabels=[])
        if t == 0:
            ax.set_ylabel('Feature ' + str(f))
        plt.imshow(sw[t,f].numpy(),cmap='gray')
        if f == 3:
            ax = plt.subplot(5, 5, plt_idx + 5)
            plt.setp(ax, xticklabels=[])
            plt.setp(ax, yticklabels=[])
            if t == 0:
                ax.set_ylabel('Sum')
            ax.set_xlabel('t = ' + str(t))
            plt.imshow(sw[t].sum(dim=0).numpy(),cmap='gray')
plt.show()


# In[19]:


len(dataset)


# In[20]:


plt.imshow(sw[0].sum(dim=0).numpy(),cmap='gray')


# In[21]:


img = sw[0].sum(dim=0).numpy()


# In[22]:


plt.imshow(img, cmap='gray')


# In[23]:


for i in range(0, len(dataset)):
    img = dataset[i][0]
    plt.imsave('image_Class_Zero'+str(i)+'.jpg', img[0].sum(dim=0).numpy(), cmap = 'gray')


# In[24]:


img = sw[0].sum(dim=0).numpy()
plt.imshow(img, cmap='gray')


# In[46]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


new_image_gauss = cv2.GaussianBlur(img, (5, 5),0)
plt.figure(figsize=(11,6))
plt.subplot(121), plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(new_image_gauss, cmap='gray'),plt.title('Gaussian Filter')
plt.xticks([]), plt.yticks([])
plt.show()


# In[48]:


new_image = cv2.blur(img,(9, 9))
plt.figure(figsize=(11,6))
plt.subplot(121), plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(new_image, cmap='gray'),plt.title('Mean filter')
plt.xticks([]), plt.yticks([])
plt.show()


# In[49]:


for i in range(0, len(dataset)):
    img = dataset[i][0]
    image = img[0].sum(dim=0).numpy()
    new_image_gauss = cv2.GaussianBlur(image, (9, 9),0)
    plt.imsave('image_Class_One_gauss'+str(i)+'.jpg', new_image_gauss, cmap = 'gray')




