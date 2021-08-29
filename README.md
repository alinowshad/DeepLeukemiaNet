# An Efficient Approach for Segmentation and Classification of Acute Lymphoblastic Leukemia via Optimized Spike-based Network

This repository contains links to all code and data for the paper:

"An Efficient Approach for Segmentation and Classification of Acute Lymphoblastic Leukemia via Optimized Spike-based Network" by Ali Noshad and Saeed Fallahi
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Examples of the images: healthy blood (a), blood with ALL blasts (b). (a1-2) and (b1-4) are zoomed subplots of the (a) and (b) images centered on lymphocytes and lymphoblasts, respectively.

![Fig1](https://user-images.githubusercontent.com/37798588/131260694-ec8408db-2f8c-41dc-880b-154a6111cdbc.PNG)

The samples taken from C-NMC blood smear microscopic images. Normal B-lymphoid precursors (a) and leukemic B-lymphoblast cells (b).

![Fig2](https://user-images.githubusercontent.com/37798588/131260703-69965468-f984-4971-9f44-d84b758eeccb.jpg)


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Source Code:

The entire pipeline consists of three phases:

## 1) Applying Image Segmentation using Spike-based Network

This part was done using a modified version of the network architecture proposed in the study of "M. Mozafari, S. R. Kheradpisheh, T. Masquelier, A. Nowzari-Dalini and M.        Ganjtabesh, "First-Spike-Based Visual Categorization Using Reward-Modulated STDP," in IEEE Transactions on Neural Networks and Learning Systems, vol. 29, no. 12, pp. 6178-         6190,      Dec. 2018, doi: 10.1109/TNNLS.2018.2826721", modified and the proposed version of the code can be found in the following address:
      
      ### DeepLeukemiaNet/Spike-based Segmentation/Spike-based Segmentation.py

The orgianl model proposed and source code in the Mozafari et al., research can be found in the following address: https://github.com/miladmozafari/SpykeTorch

The following Figure shows a sample of blood image before and after image segmentation using Spike-based network (Example of the image before and after preprocessing: (a) not healthy subject with blast cells; (b) processed image with proposed segmentation approach).

![Fig 3](https://user-images.githubusercontent.com/37798588/131241475-b803ef6d-fe51-4721-ab83-403e8acadb3a.PNG)

On the next step, we have used the Gaussian Blur technique to remove the noise of images. Gaussian blurring is the result of blurring an image by a Gaussian function. The formula of a Gaussian function in one dimension is:

![Capture](https://user-images.githubusercontent.com/37798588/131253132-7f0dc8dc-6da7-4fe7-aecb-0f9a8a24fac4.PNG)

Normalization was performed on the dataset using the following equation:

![Capture](https://user-images.githubusercontent.com/37798588/131253216-43e85d6e-39c0-4333-83c9-d8113c74985d.PNG)

All of the images were resized:

 ###### ALL-IDB: 2592 * 1944 to 224 * 224
 ###### C-NMC: 450 * 450 to 100 * 100

## 2) Data Augmentation and Principle Component Analysis (PCA):

In this study, the following transformations are used for data augmentation: the images are flipped horizontally, random width shift and height shift is performed by 10% along width and height of the images, random zoomed by 20% and random rotation of images by 30 degrees is also performed to increase the images. 

Principle Component Analysis (PCA) technique is used for the purpose of image dimensions reduction. In this technique, by identification of covariance, the desired dimensions are reduced. In the current study, the PCA technique was implemented on the gray scale images to reduce the dimensions down to 1200, and then the images were reconstructed using the reduced selected features with more than 99% explained variance. The same procedures were applied to the second dataset (C-NMC), in which the dimensions were reduced to 500 from 10K, and then the images were reconstructed using these selected features.

      ### DeepLeukemiaNet/Data Augmentation and PCA//Data-Augmentatio n & PCA.py
      Note: This source code was used for both datasets (ALL-IDB & C-CNMC)

The block diagram of the proposed preprocessing framework is shown in the following Figure:

![Fig4](https://user-images.githubusercontent.com/37798588/131253249-c5396265-b3a7-4bfe-8b58-810b455d6394.PNG)

## 3) Model Archtecture and Classification:

In this section, we developed a 31-layer 2D convolutional neural network for classification of Leukemia in blood slides. It takes a preprocessed blood slide image as input and output a probability indicating the presence or absence of Leukemia in blood slide. The model proposed here is based on the modified version of the residual learning techniques, which optimization is significantly easier and can lead to an effective training process.

The following figure shows the structure of Identity Block:

![Fig5](https://user-images.githubusercontent.com/37798588/131253408-a37f9f95-7b43-4f4d-abec-a4cea85377cd.PNG)

The following figure shows the structure of Convolutional Block:

![Fig6](https://user-images.githubusercontent.com/37798588/131253432-90fb8a7d-bc6c-48db-b551-40256ca2e852.PNG)

The followng figure illustrates the architecture of the proposed model:

![Fig7](https://user-images.githubusercontent.com/37798588/131260069-89465e37-5df2-4fc1-8973-714d9c23e958.png)

            ### DeepLeukemiaNet/ALL-IDB/ALL-IDB - Experiments.py (ALL-IDB)
            ### DeepLeukemiaNet/C-NMC/C-NMC-Experiments.py (C-NMC)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Data

The datasets used in this study can be downloaded here:

[1] https://homes.di.unimi.it/scotti/all/

[2] https://wiki.cancerimagingarchive.net/display/Public/C_NMC_2019+Dataset%3A+ALL+Challenge+dataset+of+ISBI+2019
