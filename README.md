# An Efficient Approach for Segmentation and Classification of Acute Lymphoblastic Leukemia via Optimized Spike-based Network

This repository contains links to all code and data for the paper:

"An Efficient Approach for Segmentation and Classification of Acute Lymphoblastic Leukemia via Optimized Spike-based Network" by Ali Noshad and Saeed Fallahi

--------------------------------------------------------------------------------------------------------------------------------------------------------------

Source Code:

The entire pipeline consists of three phases:

## 1) Applying Image Segmentation using Spike-based Network

This part was done using a modified version of the network architecture proposed in the study of "M. Mozafari, S. R. Kheradpisheh, T. Masquelier, A. Nowzari-Dalini and M.        Ganjtabesh, "First-Spike-Based Visual Categorization Using Reward-Modulated STDP," in IEEE Transactions on Neural Networks and Learning Systems, vol. 29, no. 12, pp. 6178-         6190,      Dec. 2018, doi: 10.1109/TNNLS.2018.2826721", modified and the proposed version of the code can be found in the following address:
      
      ### DeepLeukemiaNet/Spike-based Segmentation/Spike-based Segmentation.py

The orgianl model proposed and source code in the Mozafari et al., research can be found in the following address: https://github.com/miladmozafari/SpykeTorch

The following Figure shows a sample of blood image before and after image segmentation using Spike-based network (Example of the image before and after preprocessing: (a) not healthy subject with blast cells; (b) processed image with proposed segmentation approach).

![Fig 3](https://user-images.githubusercontent.com/37798588/131241475-b803ef6d-fe51-4721-ab83-403e8acadb3a.PNG)

All of the images were resized:

 ###### ALL-IDB: 2592 * 1944 to 224 * 224
 ###### C-NMC: 450 * 450 to 100 * 100

## 2) Data Augmentation and Principle Component Analysis (PCA):

In this study, the following transformations are used for data augmentation: the images are flipped horizontally, random width shift and height shift is performed by 10% along width and height of the images, random zoomed by 20% and random rotation of images by 30 degrees is also performed to increase the images. 


