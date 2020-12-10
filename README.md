# DeepLeukemiaNet
A 31-layer 2D convolutional neural network for classification of Leukemia in blood slides.

Development of Classification Model for Acute Lymphoblastic Leukemia Detection using Residual technique It was found in early studies that the common feature selection methods are often misdiagnosed lymphocytes (healthy) with lymphoblasts due to the morphological diversity of cells, and they cannot capture the specific features accurately and with high sensitivity. Motivated by this, in this study a robust deep residual learning model was proposed for the detection of Acute Lymphoblastic Leukemia. We developed a DeepLeukemiaNet, a 31-layer 2D convolutional neural network for classification of Leukemia in blood slides. A convolutional neural network (CNN) considered to be a deep learning approach which designed to process image data, a 2D convolutional neural networks are perform well in handling the image data.
DeepLeukemiaNet takes as inputs a preprocessed blood slide image and output a probability indicating the presence or absence of Leukemia in blood slides.

In this study, microscopic image of blood samples from Acute Lymphoblastic Leukemia Image Database namely ALL-IDB were used for the diagnosis of Leukemia. Blood samples dataset was proposed by the [1], specifically designed for the evaluation and the comparison of algorithms for segmentation and classification. The images of the dataset have been captured with an optical laboratory microscope coupled with a Canon PowerShot G5 camera. All images in the datasets are in JPG format with 24-bit color depth, and a native resolution equal to 2592 * 1944. The images related to different magnifications of the microscope (ranging from 300 to 500). The ALL-IDB database has two distinct versions (ALL-IDB1 and ALL-IDB2) focused on segmentation and classification, which can be freely used from [2].

[1] Labati, R.D., Piuri, V., Scotti, F., 2011. All-idb: The acute lymphoblastic leukemia image database for image processing. In: 18th IEEE International Conference on Image Processing, ICIP, pp. 2045–2048.

[2] [dataset] https://homes.di.unimi.it/scotti/all/
