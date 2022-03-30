# A new hybrid Framework based on Deep Neural Networks and JAYA Optimization Algorithm for Feature Selection Using SVM Applied to Classification of Acute Lymphoblastic Leukemia

This repository contains links to all code and data for the paper:

#### "A new hybrid Framework based on Deep Neural Networks and JAYA Optimization Algorithm for Feature Selection Using SVM Applied to Classification of Acute Lymphoblastic Leukemia" by Ali Noshad and Saeed Fallahi
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Examples of the images: healthy blood (a), blood with ALL blasts (b). (a1-2) and (b1-4) are zoomed subplots of the (a) and (b) images centered on lymphocytes and lymphoblasts, respectively.

![Fig1](https://user-images.githubusercontent.com/37798588/131260694-ec8408db-2f8c-41dc-880b-154a6111cdbc.PNG)

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

 ###### ALL-IDB: 2592 * 1944 to 100 * 100
 ###### C-NMC: 450 * 450 to 100 * 100

## 2) Data Augmentation & Principle Component Analysis (PCA):

In this study, the following transformations are used for data augmentation: the images are flipped horizontally, random width shift and height shift is performed by 10% along width and height of the images, random zoomed by 20% and random rotation of images by 30 degrees is also performed to increase the images. 

Principle Component Analysis (PCA) technique is used for the purpose of image dimensionality reduction. The PCA technique is implemented on the gray scale images with a feature vector of 100 × 100 × 1 = 10000 to reduce the dImensions down to 1747, and then the images are reconstructed using the reduced selected features with more than 99% of explained variance. The same procedures are applied to the second dataset, in which the dimensions are reduced to 2066 from a feature vector of 100 × 100 × 1 = 10000, and then the images are reconstructed using these extracted features.

      ### DeepLeukemiaNet/DataAugmentation_PCA_Pre-training(ResNet)_FeatureExtraction/PCA-Pretraining-FeatureExtraction-ResNet-ALL-IDB2.ipynb
      ### DeepLeukemiaNet/DataAugmentation_PCA_Pre-training(ResNet)_FeatureExtraction/PCA-Pretraining-FeatureExtraction-ResNet-C-NMC.ipynb
      Note: The output of the above codes are placed in the folders.
      ### Output (C-NMC): DeepLeukemiaNet/DataAugmentation_PCA_Pre-training(ResNet)_FeatureExtraction/OutputData(C-NMC)/
      ### Output (ALL-IDB): DeepLeukemiaNet/DataAugmentation_PCA_Pre-training(ResNet)_FeatureExtraction/OutputData(ALL-IDB2)/

The block diagram of the proposed preprocessing framework is shown in the following Figure:

![Fig4](https://user-images.githubusercontent.com/37798588/131253249-c5396265-b3a7-4bfe-8b58-810b455d6394.PNG)

## 3) ResNet Model Archtecture (Pre-training & Feature Extraction):

In this section, we developed a 31-layer 2D convolutional neural network for extracting the features from Leukemia in blood slides. In this work, the features are extracted using a convolutional neural network, based on the modified version of the residual learning techniques, which optimization is significantly easier and can lead to an effective training process.

Eventually two outputs are considered, a Softmax function is applied to output of this layer to obtain the final probability on the pre-training stage, and then Flatten layer is regarded as output for feature extraction stage.

The following figure shows the structure of Identity Block:

![Fig5](https://user-images.githubusercontent.com/37798588/131253408-a37f9f95-7b43-4f4d-abec-a4cea85377cd.PNG)

The following figure shows the structure of Convolutional Block:

![Fig6](https://user-images.githubusercontent.com/37798588/131253432-90fb8a7d-bc6c-48db-b551-40256ca2e852.PNG)

The followng figure illustrates the architecture of the proposed model:

![Fig7](https://user-images.githubusercontent.com/37798588/154528506-c0a511c7-ef05-4929-875f-e486215148e8.PNG)

            ### DeepLeukemiaNet/DataAugmentation_PCA_Pre-training(ResNet)_FeatureExtraction/PCA-Pretraining-FeatureExtraction-ResNet-ALL-IDB2.ipynb
            ### DeepLeukemiaNet/DataAugmentation_PCA_Pre-training(ResNet)_FeatureExtraction/PCA-Pretraining-FeatureExtraction-ResNet-C-NMC.ipynb
 
 ## 4) Feature Selection (JAYA):
 
In the proposed feature selection method, JAYA algorithm is used for selecting the proper features from the irrelevant or redundant features inorder to minimize the number of used features and maximize the model performance simultaneously.

            ### DeepLeukemiaNet/FeatureSelection(Jaya)/ALL-IDB/
            ### DeepLeukemiaNet/FeatureSelection(Jaya)/C-NMC/
            
![Fig10](https://user-images.githubusercontent.com/37798588/154532258-4a120e45-d2cc-4c98-be6b-3c4a2019eaf7.PNG)

Features set is retrieved using a pre-trained ResNet, and then fed into JAYA. In the next layer, the binary JAYA algorithm is applied to the new dataset composed of the features selected by the pre-trained ResNet, helps further reduce the number of features and improving the performance of the model. To demonstrate our method’s efficiency, JAYA is executed in 10 independent runs to generate 10 distinct feature sets.

## 5) Classification (SVM):

A Support Vector Machine (SVM) is a supervised machine learning algorithm that is used for learning separating functions to classify data into two categories. In this study, the RBF kernel function is used for the SVM classifier which is well known to provide excellent performance in analysing higher dimensional classification data. The overall structure of presented approach is shown in the following figure:


![Fig8](https://user-images.githubusercontent.com/37798588/154533302-7552932d-9c89-4f66-98ef-b2e8bc407b06.PNG)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
###### Preprint:
you can find a preprint version of research in the following link:

###### [Preprint] https://www.researchsquare.com/article/rs-575580/v1

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Data

The datasets used in this study can be downloaded here:

[1] https://homes.di.unimi.it/scotti/all/

[2] https://wiki.cancerimagingarchive.net/display/Public/C_NMC_2019+Dataset%3A+ALL+Challenge+dataset+of+ISBI+2019
