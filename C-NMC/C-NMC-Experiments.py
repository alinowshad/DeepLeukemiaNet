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
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# In[2]:


blood_cell_directory = os.path.join('', '../input/cnmc-leukemia/')

# creating a directory for all images present with us and bringing them under same directory
image_directory = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(blood_cell_directory, '*', '*.jpg'))}


# In[3]:


NotHealthyPaths = []
for dirname, _, filenames in os.walk(os.path.join(blood_cell_directory, 'Processed ALL')):
    for filename in filenames:
        if (filename[-3:] == 'jpg'):
                NotHealthyPaths.append(os.path.join(dirname, filename))


# In[4]:


HealthyPaths = []
for dirname, _, filenames in os.walk(os.path.join(blood_cell_directory, 'Processed Hem')):
    for filename in filenames:
        if (filename[-3:] == 'jpg'):
                HealthyPaths.append(os.path.join(dirname, filename))


# In[5]:


df = pd.DataFrame(columns = ['path', 'label'])


# In[6]:


# labels for Healthy:0, NotHealthy:1

for index1 in HealthyPaths:
    df=df.append({'path' : str(index1) , 'label' : 0} , ignore_index=True)
for index2 in NotHealthyPaths:
    df=df.append({'path' : str(index2) , 'label' : 1} , ignore_index=True)


# In[7]:


df.tail()


# In[8]:


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = r
    
    return gray


# In[9]:


# Resizing images to 224 x 224
df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,100))))


# In[10]:


for index in range(len(df['image'])):
    img = Image.fromarray(df['image'][index])
    if img.mode == 'RGB':
        df['image'][index] = rgb2gray(df['image'][index])


# In[11]:


import cv2

for i in range(len(df['image'])):
    gray = df['image'][i]
    df['image'][i] = cv2.merge([gray,gray,gray])


# In[12]:


df.head()


# In[13]:


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


# In[15]:


data = np.asarray(df['image'].tolist())

data_y = df['label']

data = data.reshape(-1,30000)


# In[16]:


df_cifar = pd.DataFrame(data)


# In[17]:


df_cifar['label'] = df['label']
print('Size of the dataframe: {}'.format(df_cifar.shape))


# In[18]:


from sklearn.decomposition import PCA
pca = PCA(500)
lower_dimension_data = pca.fit_transform(df_cifar.iloc[:,:-1])
lower_dimension_data.shape


# In[19]:


#Project lower dimension data onto original features
approximation = pca.inverse_transform(lower_dimension_data)
print(approximation.shape)
approximation = approximation.reshape(-1,100,100, 3)
X_norm = data
X_norm = X_norm.reshape(-1,100,100, 3)


# In[20]:


data = approximation


# In[21]:


train_df, test_df = train_test_split(data_y, test_size=0.2, random_state=42)


# In[22]:


y_train = to_categorical(train_df, num_classes = num_classes)
y_test = to_categorical(test_df, num_classes = num_classes)


# In[23]:


train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)


# In[24]:


# Print no. of instances of each class present to balace the data
df['label'].value_counts()


# In[25]:


features=df.drop(columns=['label'],axis=1)
target=df['label']


# In[26]:


x_train = train_df
x_test = test_df


# In[27]:


# Normalizing the data
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std


# In[28]:


del x_train_mean
del x_train_std
del x_test_mean
del x_test_std
del data
del approximation
del df
del X_norm
del lower_dimension_data


# In[29]:


x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 3)


# In[30]:


x_train.shape


# In[32]:


def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. We'll need this later to add back to the main path. 
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# In[33]:


def convolutional_block(X, f, filters, stage, block, s = 2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    
    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


# In[34]:


def ResNet(input_shape = (64, 64, 3), classes = 2):   
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [32, 32, 64], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [32, 32, 64], stage=2, block='b')
    X = identity_block(X, 3, [32, 32, 64], stage=2, block='c')
    # Stage 3
    X = convolutional_block(X, f = 3, filters = [64, 64, 128], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 128], stage=3, block='b')
    X = identity_block(X, 3, [64, 64, 128], stage=3, block='c')
    
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='c')
    # AVGPOOL.
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


# In[35]:


model = ResNet(input_shape = (100, 100, 3), classes = 2)


# In[36]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[37]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 5, verbose=1,factor=0.3, min_lr=0.000001)
x_train.shape


# In[38]:


from sklearn.model_selection import KFold
num_folds = 4
# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []
model_history = []
kfold = KFold(n_splits=num_folds, shuffle=True)


# In[39]:


# Merge inputs and targets
inputs = np.concatenate((x_train, x_validate), axis=0)
targets = np.concatenate((y_train, y_validate), axis=0)


# In[40]:


tprs = []
aucs = []
fprs = []


# In[46]:


# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    
    model = None
    model = ResNet(input_shape = (100, 100, 3), classes = 2)
  # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



  # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

  # Fit data to model
    history = model.fit(inputs[train], targets[train],
              batch_size=32,
              epochs=10,
              verbose=1,
              callbacks=[learning_rate_reduction]
        )
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    predicted = model.predict(inputs[test])
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.argmax(targets[test], axis = 1), np.argmax(predicted, axis = 1))
    fprs.append(fpr_keras)
    tprs.append(tpr_keras)
    auc_fold = auc(fpr_keras, tpr_keras)
    aucs.append(auc_fold)
    plt.plot(fpr_keras, tpr_keras, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (fold_no, auc_fold))
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    model_history.append(history)
    # Increase fold number
    fold_no = fold_no + 1


# In[47]:


print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")


# In[48]:


plt.title('Train Accuracy')
plt.plot(model_history[0].history['accuracy'], label='Train Accuracy Fold 1', color='black')
plt.plot(model_history[1].history['accuracy'], label='Train Accuracy Fold 2', color='red', )
plt.plot(model_history[2].history['accuracy'], label='Train Accuracy Fold 3', color='green', )
plt.plot(model_history[3].history['accuracy'], label='Train Accuracy Fold 4', color='blue', )
plt.legend()
plt.show()


# In[65]:


from itertools import islice 
fpr_fold = list(islice(fprs, 8, 12))
print("Fprs_Folds:", fpr_fold)


# In[66]:


tpr_fold = list(islice(tprs, 8, 12))
print("Folds:", tpr_fold)


# In[67]:


auc_fold = list(islice(aucs, 8, 12))
print("Folds:", auc_fold)


# In[76]:


plt.figure(figsize=(8,8))
plt.plot(fpr_fold[0], tpr_fold[0], 'g-o', lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (1, auc_fold[0]))
plt.plot(fpr_fold[1], tpr_fold[1], 'r-o', lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (2, auc_fold[1]))
plt.plot(fpr_fold[2], tpr_fold[2], 'b-o', lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (3, auc_fold[2]))
plt.plot(fpr_fold[3], tpr_fold[3], 'y-o', lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (4, auc_fold[3]))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
mean_tpr = np.mean(tpr_fold, axis=0)
mean_fpr = np.mean(fpr_fold, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Cross-Validation ROC of Deep Leukemia-Net',fontsize=18)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()


# In[77]:


y_pred_keras = model.predict(x_test)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.argmax(y_test, axis = 1), np.argmax(y_pred_keras, axis = 1))


# In[79]:


for i in y_pred_keras:
    print(i)


# In[80]:


for i in y_test:
    print(i)


# In[81]:


auc_keras = auc(fpr_keras, tpr_keras)


# In[82]:


auc_keras


# In[83]:


import scipy.stats
from scipy import stats


def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


alpha = .95
y_pred = np.argmax(y_pred_keras, axis = 1)
y_true = np.argmax(y_test, axis=1)

auc, auc_cov = delong_roc_variance(y_true, y_pred)

auc_std = np.sqrt(auc_cov)
lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

ci = stats.norm.ppf(
    lower_upper_q,
    loc=auc,
    scale=auc_std)

ci[ci > 1] = 1

print('AUC:', auc)
print('AUC COV:', auc_cov)
print('95% AUC CI:', ci)


# In[84]:


# Creating a function to report confusion metrics
def confusion_metrics (conf_matrix):
# save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate mis-classification
    conf_misclassification = 1- conf_accuracy
    
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    
    # calculate precision
    conf_precision = (TN / float(TN + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    print('-'*50)
    print(f'Accuracy: {round(conf_accuracy,2)}') 
    print(f'Mis-Classification: {round(conf_misclassification,2)}') 
    print(f'Sensitivity: {round(conf_sensitivity,2)}') 
    print(f'Specificity: {round(conf_specificity,2)}') 
    print(f'Precision: {round(conf_precision,2)}')
    print(f'f_1 Score: {round(conf_f1,2)}')


# In[85]:


from sklearn.metrics import classification_report
test_pred = model.predict(x_test)
print(classification_report(np.argmax(y_test, axis = 1),np.argmax(test_pred, axis = 1)))


# In[86]:


for i in test_pred:
    print(i)


# In[87]:


for i in y_test:
    print(i)


# In[88]:


test_cm = confusion_matrix(np.argmax(y_test, axis = 1),np.argmax(test_pred, axis = 1))


# In[89]:


confusion_metrics(test_cm)


# In[90]:


predict = []
cmKfold = []
for train, test in kfold.split(inputs, targets):
    pred = model.predict(inputs[test])
    predict.append(pred)
    cmkfold = confusion_matrix(np.argmax(targets[test], axis = 1),np.argmax(pred, axis = 1))
    cmKfold.append(cmkfold)


# In[91]:


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# In[92]:


pred = model.predict(x_validate)
print(classification_report(np.argmax(y_validate, axis = 1),np.argmax(pred, axis = 1)))


# In[93]:


cm = confusion_matrix(np.argmax(y_validate, axis = 1),np.argmax(pred, axis = 1))
cm


# In[102]:


cm = confusion_matrix(np.argmax(y_test, axis = 1),np.argmax(test_pred, axis = 1))
cm


# In[94]:


cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])


# In[95]:


print(cmKfold[0])
print(cmKfold[1])
print(cmKfold[2])
print(cmKfold[3])
