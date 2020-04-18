# Text Classification and Polarity Detection
Text classification for UtaPass and KKBOX total reviews using different machine learning models.

## Introduction
This text analysis is based on reviews data of UtaPass and KKBOX from Google Play platform. As a KKStreamer at KKBOX, I become more interested in Natural Language Processing, especially text classification. First, I start crawling the text data using web crawler technique, namely BeautifulSoup and Selenium. Second, I develop several different neural network architectures, including simple RNN, LSTM, GRU, and CNN, to detect the polarity of reviews from customers.

## Data Source
1. [UtaPass reviews on Google Play](https://play.google.com/store/apps/details?id=com.kddi.android.UtaPass&hl=ja&showAllReviews=true)
2. [KKBOX reviews on Google Play](https://play.google.com/store/apps/details?id=com.skysoft.kkbox.android&hl=ja&showAllReviews=true)

## Bottleneck
* Do every reviews have sentiment words or charateristic of polarity?
* Is text pre-processing (e.g. remove stop words, remove punctuation, remove bad characters) neccessary? 
* Is there any useless , redundant or invalid information about the reviews? 
* Does this dataset exist a imbalance problem?

## Flow Chart of Text Classification
![FlowChart](https://github.com/penguinwang96825/Text-Classifier-for-UtaPass-and-KKBOX/blob/master/image/flowChart.jpg)

## Information of my computer
Full Specs: 
* Processor: Intel Core i9-9900K
* Motherboard: Gigabyte Z390 AORUS MASTER
* GPU: MSI RTX2080Ti Gaming X Trio 11G
* RAM: Kingston 16GB DDR4-3000 HyperX Predator
* CPU Cooler: MasterLiquid ML240L
* Storage: Samsung SSD 970 EVO 250G (M.2 PCIe 2280) + Crucial MX500 500GB
* Power: Antec HCG750 Gold
* Case: Fractal Design R6-BKO-TG

## Using GPU instead of CPU

### Create a conda environment
1. Install python version 3.7.3: https://www.python.org/downloads/release/python-373/
2. Install Anaconda 3 for win10: https://www.anaconda.com/distribution/#download-section
3. Create a virtual environment and change the PYTHONPATH of an ipython kernel: 
* `conda update conda`
* `conda create --name my_env python=3.7.3`
* `conda activate my_env`
* `conda install ipykernel -y`
* `python -m ipykernel install --user --name my_env --display-name "My Env"`
4. GPU support software requirements:
* NVIDIA® GPU drivers: https://www.nvidia.com/Download/index.aspx?lang=en-us
* CUDA® Toolkit: https://developer.nvidia.com/cuda-toolkit-archive
* cuDNN SDK: https://developer.nvidia.com/cudnn
* (Optional) TensorRT 5.0: https://developer.nvidia.com/tensorrt
5. Windows setup
* Add the CUDA, CUPTI, and cuDNN installation directories to the `%PATH%` environmental variable. For example, if the CUDA Toolkit is installed to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0` and cuDNN to `C:\tools\cuda`, update your `%PATH%` to match:
```
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH%
```
* Add the absolute path to the TensorRTlib directory to the environment variable LD_LIBRARY_PATH

## Check whether GPU is working
1. Choose which GPU you want to use
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```
2. Check what all devices are used by tensorflow
```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
3. Check using Keras backend function
```python
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
```
4. How to get the nvidia driver version from the command line?
* Open the terminal and type in `cd C:\Program Files\NVIDIA Corporation\NVSMI` and input `nvidia-smi`
```console
C:\Users\YangWang>cd C:\Program Files\NVIDIA Corporation\NVSMI

C:\Program Files\NVIDIA Corporation\NVSMI>nvidia-smi
Sun Jun 16 03:32:53 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 430.86       Driver Version: 430.86       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1060   WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   54C    P8     6W /  N/A |     85MiB /  3072MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Preparation
1. Preparing [selenium](https://pypi.org/project/selenium/), [beautiful soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/), and [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html).
* Selenium: Selenium is an open source tool which is used for automating the tests carried out on web browsers (Web applications are tested using any web browser).
* Beautiful Soup: Beautiful Soup is a Python library for pulling data out of HTML and XML files.
* Pandas: Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
```python
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.proxy import *
from selenium.webdriver.common.by import By
import time, sys, io
import pandas as pd
```
  Note: 
  * If can not `pip install fasttext`, than take [Microsoft Visual C++ 14.0 is required (Unable to find vcvarsall.bat)](https://stackoverflow.com/questions/29846087/microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat) as a reference.
  ![Build Tools for Visual Studio](https://i.stack.imgur.com/7rK61.jpg)
  ![Build Tools for Visual Studio](https://developercommunity.visualstudio.com/storage/temp/52606-buildtools.png)
  
  * Installing MeCab on windows 10: [Windows で pip で mecab-python をいれる](https://qiita.com/yukinoi/items/990b6933d9f21ba0fb43)

2. Doing text pre-processing after installing [MeCab](https://pypi.org/project/mecab-python-windows/), [neologdn](https://pypi.org/project/neologdn/), [re](https://docs.python.org/3.6/library/re.html), and [emoji](https://pypi.org/project/emoji/)
* MeCab: MeCab is an open-source text segmentation library for use with text written in the Japanese language originally developed by the Nara Institute of Science and Technology.
* Neologdn: Neologdn is a Japanese text normalizer for mecab-neologd.
* re: A regular expression (or RE) specifies a set of strings that matches it; the functions in this module let you check if a particular string matches a given regular expression (or if a given regular expression matches a particular string, which comes down to the same thing).
* emoji: The entire set of Emoji codes as defined by the unicode consortium is supported in addition to a bunch of aliases.
```python
import MeCab
from os import path
import neologdn
import re
import emoji

pos_list = [10, 11, 31, 32, 34]
pos_list.extend(list(range(36,50)))
pos_list.extend([59, 60, 62, 67])

def create_mecab_list(text):
    mecab_list = []
    mecab = MeCab.Tagger("-Ochasen")
    mecab.parse("")
    # encoding = text.encode('utf-8')
    node = mecab.parseToNode(text)
    while node:
        if len(node.surface) > 1:
            if node.posid in pos_list:
                morpheme = node.surface
                mecab_list.append(morpheme)
        node = node.next
    return mecab_list

def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    cleaned_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return cleaned_text

def clean_text(text):
    text = give_emoji_free_text(text)
    text = neologdn.normalize(text)
    text = create_mecab_list(text)    
    return text
```
3. Since the page at Google Play has to scroll down and click the "see more" button to view the whole reviews, I have to set a function to cope with these problems.
```python
def check_exists_by_xpath(xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True

def replace_value_with_definition(key_to_find, definition):
    for key in temp.keys():
        if key == key_to_find:
            temp[key] = definition
            
def scrollDownPage(pages):
	for i in range(1,pages):
	    try:
	        # Scroll to load other reviews
	        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
	        time.sleep(1)
	        if check_exists_by_xpath('.//span[@class = "RveJvd snByac"]'):
	            driver.find_element_by_xpath('.//span[@class = "RveJvd snByac"]').click()
	            time.sleep(2)
	    except:
	        pass
```
4. Start crawling the web (Reference: https://github.com/ranjeet867/google-play-crawler)
```python
def openGooglePlayStore():
	wait = WebDriverWait(driver, 10)
	url = "https://play.google.com/store/apps/details?id=com.kddi.android.UtaPass&hl=ja&showAllReviews=true"
	driver.get(url)
	time.sleep(5)
  
def getReviewerName():
	app_user = []
	user_name = driver.find_elements_by_css_selector("span.X43Kjb")
	for n in user_name:
		app_user.append(n.text)
	return app_user

def getReviewerTime():
	app_time = []
	reviewer_time = driver.find_elements_by_css_selector("span.p2TkOb")
	for t in reviewer_time:
		app_time.append(t.text)
	return app_time

def getReviewerRating():
	app_rating = []
	reviewer_rating = driver.find_elements_by_css_selector("span.nt2C1d div.pf5lIe div[aria-label]")
	for a in reviewer_rating:
		app_rating.append(a.get_attribute( "aria-label" ))
	return app_rating

def ReviewerRating2Digits( app_rating ):
	# Transfer reviewer ratings into digits
	rating = []
	for element in app_rating:
		temp = element.split('/')[0]
		temp2 = temp.split('星 ')[1]
		rating.append(int(temp2))
	return rating

def getRatingResult():
	ratings = ReviewerRating2Digits( getReviewerRating() )
	return ratings

def getReviewerComment():
	app_comment = []
	comment = driver.find_elements_by_xpath('.//span[@jsname = "bN97Pc"]')
	for c in comment:
		app_comment.append(c.text)
	return app_comment
```
5. Transforming the data into dataframe using pandas, and removing the rows which contain empty cell.
```python
def produceReviewsDictionary():
	concat_reviews_detail_dictionary = {
	    "Reviewer": getReviewerName(),
	    "Review Date": getReviewerTime(),
	    "Reviewer Rating": getRatingResult(),
	    "Comment": getReviewerComment()
	}
	return concat_reviews_detail_dictionary

def pandas2csv(concat_reviews_detail_dictionary):
	reviews_detail = pd.DataFrame(concat_reviews_detail_dictionary)
	reviews_detail.to_csv("UtaPass_Reviews.csv")

if __name__ == '__main__':
	driver = webdriver.Chrome(r"./chromedriver")
	openGooglePlayStore()
	scrollDownPage(25)

	app_user = getReviewerName()
	app_time = getReviewerTime()
	ratings = getRatingResult()
	app_comment = getReviewerComment()

	driver.quit()
	pandas2csv(produceReviewsDictionary())
```
![GitHub Logo](https://github.com/penguinwang96825/Text_Classification/blob/master/image/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-08%20%E4%B8%8B%E5%8D%883.38.26.png)
6. Finally, combine KKBOX reviews dataframe and UtaPass dataframe~ There would be 2250 reviews over two dataset.

## What is Neural Network?
### A simple concept for NN
Artificial neural networks, invented in the 1940’s, are a way of calculating an output from an input (a classification) using weighted connections (“synapses”) that are calculated from repeated iterations through training data. Each pass through the training data alters the weights such that the neural network produces the output with greater “accuracy” (lower error rate).

![NN](https://storage.googleapis.com/static.leapmind.io/blog/2017/06/bdc93d33df3826ed40e029cd8893466f.png)

Deep learning neural networks are trained using the stochastic gradient descent optimization algorithm. As part of the optimization algorithm, the error for the current state of the model must be estimated repeatedly. This requires the choice of an error function, conventionally called a loss function, that can be used to estimate the loss of the model so that the weights can be updated to reduce the loss on the next evaluation.

![Overfitting/Underfitting a Model](https://cdn-images-1.medium.com/max/1000/1*6vPGzBNppqMHllg1o_se8Q.png)
* Underfitting: a linear function is not sufficient to fit the training samples.
* Overfitting: for higher degrees the model will overfit the training data, i.e. it learns the noise of the training data.

### What can we do to cope with overfitting?
1. Takahiro Ishihara addressed these issues by applying eigendecomposition to each slice matrix of a tensor to reduce the number of parameters. [Neural Tensor Networks with Diagonal Slice Matrices](https://www.aclweb.org/anthology/N18-1047)
2. Another simple and powerful regularization technique for neural networks and deep learning models is adding Dropout layer proposed by Srivastava, et al. in their 2014 paper “[Dropout: A Simple Way to Prevent Neural Networks from Overfitting.](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)”
3. Early stopping is a method that allows you to specify an arbitrary large number of training epochs and stop training once the model performance stops improving on a hold out validation dataset.
![Early stopping](https://cdn-images-1.medium.com/max/1200/1*QckgibgJ74BhMaqinqwSDw.png)

## What is Sentiment Analysis?
### Sentiment Analysis & Topic Categorization
Sentiment analysis attempts to determine the sentiment of a text. You can formulate this problem in several ways, depending on your working definition of “sentiment” and “text.”

Sentiment can be binary, categorical, ordinal, or continuous. When modeled as continuous, sentiment is often called “polarity,” an analogue for positive and negative charges. The graphic below illustrates these different options and provides an example of each.

![Sentiment Analysis](https://cdn-images-1.medium.com/max/1400/1*P5mOEUJ_h4rahnvPQcgirA.jpeg)

## Let the rob hit the road!
1. We first start by loading the raw data. Each textual reviews is splitted into a positive part and a negative part. We group them together in order to start with only raw text data and no other information. If the reviewer rating is lower than 3 stars, we will divide it into the negative group. 
```python
df = pd.read_csv("reviews_kkstream.csv")
import numpy as np

# create the label
df["is_bad_review"] = df["Reviewer Ratings"].apply(lambda x: 0 if int(x) <= 3 else 1)
# select only relevant columns
df = df[["Review Body", "is_bad_review"]]
df.head()
```
![GitHub Logo](https://github.com/penguinwang96825/Text_Classification/blob/master/image/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-08%20%E4%B8%8B%E5%8D%884.19.52.png)

2. Split the data into training data and testing data
* Training set: a subset to train a model
* Testing set: a subset to test the trained model

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

sentences = df['Review Body'].apply(str).values
y = df['is_bad_review'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=1000)
```

3. Import the packages we need
```python
import tensorflow as tf
import numpy
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import LSTM,Bidirectional
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import Convolution1D, MaxPooling1D
from keras.engine import Input
from keras.optimizers import SGD
from keras.preprocessing import text,sequence
import pandas
import os
from gensim.models.word2vec import Word2Vec
```

4. Set all the parameters
```python
# Input parameters
max_features = 5000
max_len = 200
embedding_size = 300

# Convolution parameters
filter_length = 3
nb_filter = 150
pool_length = 2
cnn_activation = 'relu'
border_mode = 'same'

# RNN parameters
output_size = 50
rnn_activation = 'tanh'
recurrent_activation = 'hard_sigmoid'

# Compile parameters
loss = 'binary_crossentropy'
optimizer = 'rmsprop'

# Training parameters
batch_size = 128
nb_epoch = 250
validation_split = 0.25
shuffle = True
```

5. Build the word2vec model to do word embedding. (Reference: https://github.com/philipperemy/japanese-words-to-vectors/blob/master/README.md)

Training a Japanese Wikipedia Word2Vec Model by Gensim and Mecab: 
https://github.com/Kyubyong/wordvectors
https://qiita.com/omuram/items/6570973c090c6f0cb060
https://textminingonline.com/training-a-japanese-wikipedia-word2vec-model-by-gensim-and-mecab
```python
# Build vocabulary & sequences
tk = text.Tokenizer(nb_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tk.fit_on_texts(sentences)
x = tk.texts_to_sequences(sentences)
word_index = tk.word_index
x = sequence.pad_sequences(x,maxlen=max_len)

# Build pre-trained embedding layer
import gensim
w2v = Word2Vec.load('ja-gensim.50d.data.model')

from collections import Counter

word_vectors = w2v.wv
MAX_NB_WORDS = len(word_vectors.vocab)
MAX_SEQUENCE_LENGTH = 200
WV_DIM = 50
nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))
vocab = Counter()
word_index = {t[0]: i+1 for i,t in enumerate(vocab.most_common(MAX_NB_WORDS))}

# we initialize the matrix with random numbers
import numpy as np
wv_matrix = (np.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass      

import tensorflow as tf
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

embedding_layer = Embedding(nb_words, 
                     WV_DIM, 
                     mask_zero = False, 
                     weights = [wv_matrix], 
                     input_length = MAX_SEQUENCE_LENGTH, 
                     trainable = False)
```

6. Construct the five models.
Reference: [amazon-sentiment-keras-experiment](https://github.com/asanilta/amazon-sentiment-keras-experiment), [img2txt(CNN+LSTM)](https://github.com/teratsyk/bokete-ai)
* Simple RNN
```python
# Simple RNN

model_RNN = Sequential()
model_RNN.add(embedding_layer)
model_RNN.add(SimpleRNN(output_dim=output_size, activation=rnn_activation))
model_RNN.add(Dropout(0.25))
model_RNN.add(Dense(1))
model_RNN.add(Activation('sigmoid'))

model_RNN.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

print('Simple RNN')

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

path = 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(path, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'auto')
early_stopping = EarlyStopping(monitor='loss', patience = 8, verbose = 1, mode = 'auto')

history_RNN = model_RNN.fit(x, y, batch_size = batch_size, 
                            epochs = nb_epoch, 
                            validation_split = validation_split, 
                            shuffle = shuffle, 
                            verbose = 1, 
                            callbacks = [model_checkpoint, early_stopping])
```
* GRU
```python
# GRU

model_GRU = Sequential()
model_GRU.add(embedding_layer)
model_GRU.add(GRU(units = output_size, activation = rnn_activation,recurrent_activation = recurrent_activation))
model_GRU.add(Dropout(0.25))
model_GRU.add(Dense(1))
model_GRU.add(Activation('sigmoid'))

model_GRU.compile(loss = loss,
                  optimizer = optimizer,
                  metrics = ['accuracy'])

print('GRU')

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

path = 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(path, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'auto')
early_stopping = EarlyStopping(monitor='loss', patience = 8, verbose = 1, mode = 'auto')

history_GRU = model_GRU.fit(x, y, batch_size = batch_size, 
                          epochs = nb_epoch, 
                          validation_split = validation_split, 
                          shuffle = shuffle, 
                          verbose = 1, 
                          callbacks = [model_checkpoint, early_stopping])
```
* LSTM
```python
# LSTM

model_LSTM = Sequential()
model_LSTM.add(embedding_layer)
model_LSTM.add(Dropout(0.5))
model_LSTM.add(LSTM(units = output_size, activation = rnn_activation, recurrent_activation = recurrent_activation))
model_LSTM.add(Dropout(0.25))
model_LSTM.add(Dense(1))
model_LSTM.add(Activation('sigmoid'))

model_LSTM.compile(loss=loss,
                   optimizer=optimizer,
                   metrics=['accuracy'])

print('LSTM')

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

path = 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(path, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'auto')
early_stopping = EarlyStopping(monitor='loss', patience = 8, verbose = 1, mode = 'auto')

history_LSTM = model_LSTM.fit(x, y, batch_size = batch_size, 
                              epochs = nb_epoch, 
                              validation_split = validation_split, 
                              shuffle = shuffle, 
                              verbose = 1, 
                              callbacks = [model_checkpoint, early_stopping])
```
* BiLSTM
```python
# Bidirectional LSTM

model_BiLSTM = Sequential()
model_BiLSTM.add(embedding_layer)
model_BiLSTM.add(Bidirectional(LSTM(units=output_size,activation=rnn_activation,recurrent_activation=recurrent_activation)))
model_BiLSTM.add(Dropout(0.25))
model_BiLSTM.add(Dense(1))
model_BiLSTM.add(Activation('sigmoid'))

model_BiLSTM.compile(loss=loss,
                     optimizer=optimizer,
                     metrics=['accuracy'])

print('Bidirectional LSTM')

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

path = 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(path, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'auto')
early_stopping = EarlyStopping(monitor='loss', patience = 8, verbose = 1, mode = 'auto')

history_BiLSTM = model_BiLSTM.fit(x, y, batch_size = batch_size, 
                                  epochs = nb_epoch, 
                                  validation_split = validation_split, 
                                  shuffle = shuffle, 
                                  verbose = 1, 
                                  callbacks = [model_checkpoint, early_stopping])
```
* CNN + LSTM (Based on "[Convolutional Neural Networks for Sentence Classification](http://arxiv.org/pdf/1408.5882v2.pdf)" by Yoon Kim)
```python
# CNN + LSTM

model_CNN_LSTM = Sequential()
model_CNN_LSTM.add(embedding_layer)
model_CNN_LSTM.add(Dropout(0.5))
model_CNN_LSTM.add(Convolution1D(filters=nb_filter,
                        kernel_size=filter_length,
                        border_mode=border_mode,
                        activation=cnn_activation,
                        subsample_length=1))
model_CNN_LSTM.add(MaxPooling1D(pool_size=pool_length))
model_CNN_LSTM.add(LSTM(units=output_size,activation=rnn_activation,recurrent_activation=recurrent_activation))
model_CNN_LSTM.add(Dropout(0.25))
model_CNN_LSTM.add(Dense(1))
model_CNN_LSTM.add(Activation('sigmoid'))
model_CNN_LSTM.compile(loss=loss,
                       optimizer=optimizer,
                       metrics=['accuracy'])

print('CNN + LSTM')

from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

path = 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(path, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'auto')
early_stopping = EarlyStopping(monitor='loss', patience = 8, verbose = 1, mode = 'auto')

history_CNN_LSTM = model_CNN_LSTM.fit(x, y, batch_size = batch_size, 
                                      epochs = nb_epoch, 
                                      validation_split = validation_split, 
                                      shuffle = shuffle, 
                                      verbose = 1, 
                                      callbacks = [model_checkpoint, early_stopping])
```

7. Define two plot function to plot the history of accuracy and loss by using matplotlib.
```python
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history_ggplot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
def plot_history(history):
    # plot results
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.title('Loss')
    epochs = len(loss)
    plt.plot(range(epochs), loss, marker='.', label='loss')
    plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.set_facecolor('snow')
    plt.grid(color='lightgray', linestyle='-', linewidth=1)
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.subplot(2,1,2)
    plt.title('Accuracy')
    plt.plot(range(epochs), acc, marker='.', label='acc')
    plt.plot(range(epochs), val_acc, marker='.', label='val_acc')
    plt.legend(loc='best')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.set_facecolor('snow')
    plt.grid(color='lightgray', linestyle='-', linewidth=1)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()
```

8. Compare the performance among the five deep learning models.
* Simple RNN
![Simple RNN](https://github.com/penguinwang96825/Text_Classification/blob/master/image/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-08%20%E4%B8%8B%E5%8D%884.47.22.png)

* GRU
![GRU](https://github.com/penguinwang96825/Text_Classification/blob/master/image/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-08%20%E4%B8%8B%E5%8D%884.47.42.png)

* LSTM
![LSTM](https://github.com/penguinwang96825/Text_Classification/blob/master/image/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-08%20%E4%B8%8B%E5%8D%884.48.14.png)

* BiLSTM
![BiLSTM](https://github.com/penguinwang96825/Text_Classification/blob/master/image/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-08%20%E4%B8%8B%E5%8D%884.48.32.png)

* CNN + LSTM
![CNN + LSTM](https://github.com/penguinwang96825/Text_Classification/blob/master/image/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-08%20%E4%B8%8B%E5%8D%884.48.45.png)

9. In training a neural network, f1 score is an important metric to evaluate the performance of classification models, especially for unbalanced classes where the binary accuracy is useless.
```python
tk = text.Tokenizer(nb_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tk.fit_on_texts(sentences_test)
sentences_test = tk.texts_to_sequences(sentences_test)
word_index = tk.word_index
sentences_test = sequence.pad_sequences(sentences_test,maxlen=max_len)

y_pred_temp = model_CNN_LSTM.predict_classes(sentences_test).tolist()
y_pred = []
for i in y_pred_temp:
    y_pred.append(str(i).strip('[]'))
y_pred = [int(i) for i in y_pred]
print("Size of label: ", len(y_pred))

y_test_temp = y_test.tolist()
y_test = []
for i in y_test_temp:
    y_test.append(str(i).strip('[]'))
y_test = [int(i) for i in y_test]
print("Size of label: ", len(y_test))
```
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)
```

## Future Roadmap
It is completely possible to use only raw text as input for making predictions. The most important thing is to be able to extract the relevant features from this raw source of data. Although the accuracy of the models are really low and need more improvement, I have done a practice with a full harvest.

Text binary classifier is a meat-and-potatoes issue for most sentiment analysis, and there are still many things can be done on this topic. Future works might construct the multi-class text classifier to divide consumer reviews by different issue types. (Function, UI, Crash, Truncate, Subscription, Server, Enhancement, Other)
