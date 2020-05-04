# Text Classification and Polarity Detection
Text classification for UtaPass and KKBOX total reviews using different machine learning models.

## Introduction
This sentiment analysis is based on reviews data of UtaPass and KKBOX from Google Play platform. As a KKStreamer at KKBOX, I become more interested in Natural Language Processing, especially text classification. First, I start crawling the text data using web crawler technique, namely BeautifulSoup and Selenium. Second, I develop several different neural network architectures, including simple RNN, LSTM, GRU, and CNN, to detect the polarity of reviews from customers.

## Data Source
1. [UtaPass](https://play.google.com/store/apps/details?id=com.kddi.android.UtaPass&hl=ja&showAllReviews=true) reviews on Google Play
2. [KKBOX](https://play.google.com/store/apps/details?id=com.skysoft.kkbox.android&hl=ja&showAllReviews=true) reviews on Google Play

## Bottleneck
* Is text pre-processing (e.g. remove stop words, remove punctuation, remove bad characters) neccessary? 
* Is there any useless , redundant or invalid information about the reviews? 
* Do every reviews have sentiment words or charateristic of polarity?
* Does this dataset exist a imbalance problem?

## Flow Chart of Text Classification
![FlowChart](https://github.com/penguinwang96825/Text-Classifier-for-UtaPass-and-KKBOX/blob/master/image/flowChart.jpg)

## Workstation
* Processor: Intel Core i9-9900K
* Motherboard: Gigabyte Z390 AORUS MASTER
* GPU: MSI RTX2080Ti Gaming X Trio 11G
* RAM: Kingston 64GB DDR4-3000 HyperX Predator
* CPU Cooler: MasterLiquid ML240L
* Storage: PLEXTOR M9PeGN 1TB M.2 2280 PCIe SSD
* Power: Antec HCG750 Gold
* Case: Fractal Design R6-BKO-TG

## Create Conda Environment
1. Install python version 3.7.3: https://www.python.org/downloads/release/python-373/
2. Install Anaconda 3 for win10: https://www.anaconda.com/distribution/#download-section
3. Create a virtual environment and change the PYTHONPATH of an ipython kernel: 
```shell
conda update conda
conda create --name my_env python=3.7.3
conda activate my_env
conda install ipykernel -y
python -m ipykernel install --user --name my_env --display-name "My Env"
```
4. GPU support software requirements:
    * NVIDIA® GPU drivers: https://www.nvidia.com/Download/index.aspx?lang=en-us
    * CUDA® Toolkit: https://developer.nvidia.com/cuda-toolkit-archive
    * cuDNN SDK: https://developer.nvidia.com/cudnn
    * (Optional) TensorRT 5.0: https://developer.nvidia.com/tensorrt
5. Windows setup
    * Add the CUDA, CUPTI, and cuDNN installation directories to the `%PATH%` environmental variable. For example, if the CUDA Toolkit is installed to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0` and cuDNN to `C:\tools\cuda`, update your `%PATH%` to match:
```shell
$ SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;%PATH%
$ SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64;%PATH%
$ SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include;%PATH%
$ SET PATH=C:\tools\cuda\bin;%PATH%
```
* Add the absolute path to the TensorRTlib directory to the environment variable LD_LIBRARY_PATH

### Check whether GPU is working
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
```shell
$ C:\Users\YangWang>cd C:\Program Files\NVIDIA Corporation\NVSMI
```
```console
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
1. Preparing [Selenium](https://pypi.org/project/selenium/), [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/), and [Pandas](https://pandas.pydata.org/pandas-docs/stable/install.html).
* Selenium: Selenium is an open source tool used for automating.
* Beautiful Soup: BeautifulSoup is a Python library for parsing data out of HTML and XML files.
* Pandas: Pandas is an open source data analysis tools for the Python programming language.
```python
import time
import sys
import io
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.proxy import *
from selenium.webdriver.common.by import By
```
  Note: 
1. If `pip install fasttext` doesn't work, look at this [solution](https://stackoverflow.com/questions/29846087/microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat).
![Build Tools for Visual Studio](https://i.stack.imgur.com/7rK61.jpg)
![Build Tools for Visual Studio](https://developercommunity.visualstudio.com/storage/temp/52606-buildtools.png)

2. Install [MeCab](https://qiita.com/yukinoi/items/990b6933d9f21ba0fb43) on win10.
* Download [MeCab 64bit version](https://github.com/ikegami-yukino/mecab/releases/download/v0.996.2/mecab-64-0.996.2.exe) first.
* Run `pip install https://github.com/ikegami-yukino/mecab/archive/v0.996.2.tar.gz` in terminal.
* Run `python -m pip install mecab` in terminal.

3. Text pre-processing after installing [MeCab](https://pypi.org/project/mecab-python-windows/), [neologdn](https://pypi.org/project/neologdn/), [re](https://docs.python.org/3.6/library/re.html), and [emoji](https://pypi.org/project/emoji/)

* *MeCab* is an open-source tokenizer written in the Japanese developed by Nara Institute of Science and Technology.
* *Neologdn* is a Japanese text normalizer for mecab-neologd.
* *re* specifies a set of strings that matches it.
* *emoji* is listed on this [cheatsheet](https://www.webfx.com/tools/emoji-cheat-sheet/).

## Main Code for Crawling

### Data Preprocessing
```python
import os
import MeCab
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

### Scroll-down Feature and Click-button Feature
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

### Start Crawling
Reference from [Ranjeet Singh](https://github.com/ranjeet867/google-play-crawler).
```python
def open_google_play_store():
    wait = WebDriverWait(driver, 10)
    url = "https://play.google.com/store/apps/details?id=com.kddi.android.UtaPass&hl=ja&showAllReviews=true"
    driver.get(url)
    time.sleep(5)
  
def get_reviewer_name():
    app_user = []
    user_name = driver.find_elements_by_css_selector("span.X43Kjb")
    for n in user_name:
        app_user.append(n.text)
    return app_user

def get_reviewer_time():
    app_time = []
    reviewer_time = driver.find_elements_by_css_selector("span.p2TkOb")
    for t in reviewer_time:
        app_time.append(t.text)
    return app_time

def get_reviewer_rating():
    app_rating = []
    reviewer_rating = driver.find_elements_by_css_selector("span.nt2C1d div.pf5lIe div[aria-label]")
    for a in reviewer_rating:
        app_rating.append(a.get_attribute( "aria-label" ))
    return app_rating

def reviewer_rating_to_digits(app_rating):
    # Transfer reviewer ratings into digits
    rating = []
    for element in app_rating:
        temp = element.split('/')[0]
        temp2 = temp.split('星 ')[1]
        rating.append(int(temp2))
    return rating

def get_rating_result():
    ratings = ReviewerRating2Digits(getReviewerRating())
    return ratings

def get_reviewer_comment():
    app_comment = []
    comment = driver.find_elements_by_xpath('.//span[@jsname = "bN97Pc"]')
    for c in comment:
        app_comment.append(c.text)
    return app_comment
```

### Data Storage
There are 2250 reviews over two datasets (UtaPass and KKBOX). 

```python
def produce_reviews_dictionary():
    concat_dictionary = {
        "Reviewer": get_reviewer_name(),
        "Review Date": get_reviewer_time(),
        "Reviewer Rating": get_rating_result(),
        "Comment": get_reviewer_comment()
    }
    return concat_dictionary

def pandas2csv(concat_dictionary):
    reviews_detail = pd.DataFrame(concat_dictionary)
    reviews_detail.to_csv("UtaPass_Reviews.csv")
```

Take a look at the dataframe.
||Author Name|Review Date|Reviewer Ratings|Review Content|Developer Reply|
|---|---|---|---|---|---|
|195|眞也大平|2018年12月4日|1|聴い途中止まる強制終了する止まる辞め||
|13|狼音牙|2019年4月22日|1|LISMO使えなっ早くもどせうたパスLISMO使い||
|47|美能孝行|2019年3月12日|3|アルバム曲名読み方登録それ名前反映不具合かなり継続技術改善でき諦め使いあり||
|142|梅川洋子|2019年2月14日|4|いつ聴けるいい||
|45|わんたった|2019年4月27日|1|アンストアプリ残っ||

## Deep Learning Application

### Neural Network
* Feedforward neural network is an artificial neural network wherein connections between the nodes do not form a cycle.

![ANN](https://storage.googleapis.com/static.leapmind.io/blog/2017/06/bdc93d33df3826ed40e029cd8893466f.png)

* Convolutional neura network have multiple layers; including convolutional layer, non-linearity layer, pooling layer and fully-connected layer. The convolutional and fully-connected layers have parameters but pooling and non-linearity layers don't have. CNN has an excellent performance in machine learning problems. Specially the applications that deal with image data.

![CNN](https://www.mathworks.com/solutions/deep-learning/convolutional-neural-network/_jcr_content/mainParsys/band_copy_copy_14735_1026954091/mainParsys/columns_1606542234_c/2/image.adapt.full.high.jpg/1586420862596.jpg)

* Recurrent neura network is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence.

![RNN](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/RNN.png)

### Overfitting? Underfitting?

#### Definition
* Underfitting: a linear function is not sufficient to fit the training samples.
* Overfitting: for higher degrees the model will overfit the training data, i.e. it learns the noise of the training data.
![Overfitting/Underfitting](https://builtin.com/sites/default/files/styles/ckeditor_optimize/public/inline-images/Curse%20of%20Dimensionality%20overfit%20vs%20underfit.png)

#### Solution
1. Dropout layer: Srivastava, et al. proposed in their 2014 paper "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." [[paper link](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)]

2. Early Stopping: It is a form of regularization used to avoid overfitting when training a learner with an iterative method, such as gradient descent.

![Early stopping](https://cdn-images-1.medium.com/max/1200/1*QckgibgJ74BhMaqinqwSDw.png)

3. Eigendecomposition: Takahiro Ishihara applys eigendecomposition to each slice matrix of a tensor to reduce the number of parameters. [[paper link](https://www.aclweb.org/anthology/N18-1047)]


### Sentiment Analysis
Sentiment analysis is contextual mining of text which identifies and extracts subjective information in source material, and helping a business to understand the social sentiment of their brand, product or service while monitoring online conversations.

Sentiment can be various. The image below illustrates these different types of sentiment and provides examples.

![Sentiment Analysis](https://cdn-images-1.medium.com/max/1400/1*P5mOEUJ_h4rahnvPQcgirA.jpeg)

## Main Code for Modelling

### Let the rob hit the road!

#### Numerise Data
First, split dataframe into two categories: positive and negative. Second, do some text preprocessing. For instance, if rating is lower than 3 stars, label it as negative.

```python
df = pd.read_csv("reviews_kkstream.csv")
df["label"] = df["Reviewer Ratings"].apply(lambda x: 0 if int(x) <= 3 else 1)
df = df[["Review Body", "label"]]
df.columns = ["content", "label"]
df.head()
```

||content|label|
|---|---|---|
|0|歌詞見れるいいずれ誤字あるあとお気に入りプレイリスト開くライブラリ更新リセットれるマジ入れラ...|0|
|1|通知切る方法分かりすぎる見る新たアプリ入れ通知切れ判明アプリ開発若者毎日アクティブ新曲買っ聴...|0|
|2|どうしてもLISMO比べダウンロード反映LISMO動画一覧表示パス分離とにかく使いLISMO...|0|
|3|以前購入機種だぶっダウンロードれる消す出来機種するダウンロード出来有るガラケー購入スマ出来有り|0|
|4|LISMOライブラリ開けなっ愛着あっLISMO使っ消し下らないうたパスLISMOいらついて最...|0|

#### Cummulative percentage
```python
df["length"] = df["content"].map(len)
df["length"].plot.hist(bins=300, density=True, cumulative=True, histtype='step', range=(0, 110))
```

![cum](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cum.png)

```python
text_len = df["length"].values
max_len = text_len.max()

len_sum = [0] * max_len
for i in text_len:
    len_sum[i-1] += 1
    
len_cum = [len_sum[0]] + [0] * (max_len-1)
for i in range(1, max_len):
    len_cum[i] += len_sum[i] + len_cum[i-1]

print('Cumulative %   # Words  # Comments')
for i in range(max_len):
    len_cum[i] /= len(text_len)
    if len_sum[i] != 0:
        if (len_cum[i] >= 0.8 and len_cum[i-1] < 0.8):
            print(' %.5f   \t  %d \t    %d'%(len_cum[i]*100, i, len_sum[i]))
        if (len_cum[i] >= 0.85 and len_cum[i-1] < 0.85):
            print(' %.5f   \t  %d \t    %d'%(len_cum[i]*100, i, len_sum[i]))
        if (len_cum[i] >= 0.9 and len_cum[i-1] < 0.9):
            print(' %.5f   \t  %d \t    %d'%(len_cum[i]*100, i, len_sum[i]))
        if (len_cum[i] >= 0.92 and len_cum[i-1] < 0.92):
            print(' %.5f   \t  %d \t    %d'%(len_cum[i]*100, i, len_sum[i]))
        if (len_cum[i] >= 0.95 and len_cum[i-1] < 0.95):
            print(' %.5f   \t  %d \t    %d'%(len_cum[i]*100, i, len_sum[i]))
```

```shell
Cumulative %   # Words  # Comments
 80.26820         48        4
 85.72797         55        9
 90.32567         62        5
 92.14559         69        2
 95.21073         83        3
```

#### Import Packages
```python
import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import InputLayer
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Add
from keras.layers import Concatenate
from keras.layers import ZeroPadding1D
from keras.layers import Dropout
from keras.layers import SpatialDropout1D
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalMaxPool1D
from keras.layers import AveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import SimpleRNN
from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import pandas
import os
from gensim.models.word2vec import Word2Vec
from collections import Counter
```

#### Set Parameters
```python
# Input parameters
config = {
    # Text parameters
    "MAX_FEATURE": 10000, 
    "MAX_LEN": 64, 
    "EMBED_SIZE": 300, 

    # Convolution parameters
    "filter_length": 3, 
    "nb_filter": 150, 
    "pool_length": 2, 
    "cnn_activation": 'relu', 
    "border_mode": 'same', 

    # RNN parameters
    "lstm_cell": 128, 
    "output_size": 50, 
    "rnn_activation": 'tanh', 
    "recurrent_activation": 'hard_sigmoid', 
    
    # FC and Dropout
    "fc_cell": 128, 
    "dropout_rate": 0.25, 

    # Compile parameters
    "loss": 'binary_crossentropy', 
    "optimizer": 'adam', 

    # Training parameters
    "batch_size": 256, 
    "nb_epoch": 100, 
    "validation_split": 0.30, 
    "shuffle": True
}
```

#### Word Embedding
Build the word2vec model to do word embedding. [[Reference](https://github.com/philipperemy/japanese-words-to-vectors/blob/master/README.md)]

Training a Japanese Wikipedia Word2Vec Model by Gensim and Mecab: 
 * Kyubyong Park's [GitHub](https://github.com/Kyubyong/wordvectors)
 * Omuram's [Qiita](https://qiita.com/omuram/items/6570973c090c6f0cb060)
 * TextMiner's [Website](https://textminingonline.com/training-a-japanese-wikipedia-word2vec-model-by-gensim-and-mecab)
```python
# Build vocabulary & sequences
def get_preprocessed_seq(sentences, tokenizer):
    """
    input:
        sentences: numpy.ndarray
    output: 
        x: 
        x.shape: (# of sentences, sentence_max_length)
    """
    # Build vocabulary & sequences
    word_index = tokenizer.word_index

    x = tokenizer.texts_to_sequences(sentences)
    x = sequence.pad_sequences(x, maxlen=config["MAX_LEN"], padding="pre")
    
    return x

# Build pre-trained embedding matrix
def get_embedding_matrix(w2v):
    # Get word vector and load vocabulary from pretrained w2v model
    word_vectors = w2v.wv
    MAX_VOCAB = len(word_vectors.vocab)
    nb_words = min(config["MAX_FEATURE"], MAX_VOCAB)
    
    # Get word index
    counter = Counter()
    word_index = {t[0]: i+1 for i,t in enumerate(counter.most_common(MAX_VOCAB))}

    # Initialize the matrix with random numbers
    wv_matrix = (np.random.rand(nb_words, EMBED_SIZE) - 0.5) / 5.0
    for word, i in word_index.items():
        if i >= MAX_VOCAB:
            continue
        try:
            embedding_vector = word_vectors[word]
            wv_matrix[i] = embedding_vector
        except:
            pass
    print("Vocabulary size: {}\nEmbedding size: {}".format(wv_matrix.shape[0], wv_matrix.shape[1]))
    
    return wv_matrix

wv_matrix = get_embedding_matrix(w2v)
```

#### Build Tokeniser
Reference from [Japanese NLP Library](https://jprocessing.readthedocs.io/en/latest/)

```python
sentences = df['content'].apply(str).values
y = df['label'].values

tokenizer = text.Tokenizer(num_words=config["MAX_FEATURE"], lower=True, split=" ")
tokenizer.fit_on_texts(sentences)
```

#### Split Data
Split the data into training data (80%) and testing data (20%).
* Training set: a subset to train a model
* Testing set: a subset to test the trained model

```python
x_train, x_test, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=17)
```

#### Build Model
Construct neural network architectures.

Reference: 
* Asanilta Fahda's [GitHub](https://github.com/asanilta/amazon-sentiment-keras-experiment)
* teratsyk's [GitHub](https://github.com/teratsyk/bokete-ai)

Build a customised metrics function to record f1 value after each epoch.
```python
def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
```

##### Simple RNN
```python
def train_simple_rnn(x, y, wv_matrix):
    tf.keras.backend.clear_session()
    
    model = Sequential()
    model.add(Embedding(wv_matrix.shape[0], wv_matrix.shape[1], mask_zero=False, 
                        weights=[wv_matrix], input_length=config["MAX_LEN"], trainable=False))
    model.add(Dropout(config['dropout_rate']))
    model.add(SimpleRNN(units=config["output_size"], activation=config["rnn_activation"]))
    model.add(Dropout(config['dropout_rate']))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("Simple RNN: \n")
    print("="*20, "Start Training", "="*20)

    path = 'weights\{}_weights.hdf5'.format("rnn")
    model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=3, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    history = model.fit(
        x, y, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose = 1, 
        callbacks = [model_checkpoint, early_stopping, reduce_lr])
    
    return history, model
```
##### GRU
```python
def train_gru(x, y, wv_matrix):
    tf.keras.backend.clear_session()
    
    model = Sequential()
    model.add(Embedding(wv_matrix.shape[0], wv_matrix.shape[1], mask_zero=False, 
                        weights=[wv_matrix], input_length=config["MAX_LEN"], trainable=False))
    model.add(Dropout(config['dropout_rate']))
    model.add(GRU(units=config["output_size"], return_sequences=False))
    model.add(Dropout(config['dropout_rate']))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("GRU: \n")
    print("="*20, "Start Training", "="*20)

    path = 'weights\{}_weights.hdf5'.format("gru")
    model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=3, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    history = model.fit(
        x, y, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose = 1, 
        callbacks = [model_checkpoint, early_stopping, reduce_lr])
    
    return history, model
```
##### LSTM
```python
def train_lstm(x, y, wv_matrix):
    tf.keras.backend.clear_session()
    
    model = Sequential()
    model.add(Embedding(wv_matrix.shape[0], wv_matrix.shape[1], mask_zero=False, 
                        weights=[wv_matrix], input_length=config["MAX_LEN"], trainable=False))
    model.add(Dropout(config['dropout_rate']))
    model.add(CuDNNLSTM(units=config['lstm_cell'], return_sequences=False))
    model.add(Dropout(config['dropout_rate']))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("LSTM: \n")
    print("="*20, "Start Training", "="*20)

    path = 'weights\{}_weights.hdf5'.format("lstm")
    model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=3, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    history = model.fit(
        x, y, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose = 1, 
        callbacks = [model_checkpoint, early_stopping, reduce_lr])
    
    return history, model
```
##### BiLSTM
```python
def train_bilstm(x, y, wv_matrix):
    tf.keras.backend.clear_session()
    
    model = Sequential()
    model.add(Embedding(wv_matrix.shape[0], wv_matrix.shape[1], mask_zero=False, 
                        weights=[wv_matrix], input_length=config["MAX_LEN"], trainable=False))
    model.add(Bidirectional(LSTM(units=config['output_size'], return_sequences=True)))
    model.add(Bidirectional(LSTM(units=config['output_size'], return_sequences=False)))
    model.add(Dropout(config['dropout_rate']))
    model.add(Dense(config['fc_cell']))
    model.add(Dropout(config['dropout_rate']))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("Bidirectional LSTM: \n")
    print("="*20, "Start Training", "="*20)

    path = 'weights\weights.hdf5'
    model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=8, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=0.001)

    history = model.fit(
        x, y, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose = 1, 
        callbacks = [model_checkpoint, early_stopping, reduce_lr])
    
    return history, model
```
##### CNN + LSTM 
Based on "Convolutional Neural Networks for Sentence Classification" written by Yoon Kim [[paper link](http://arxiv.org/pdf/1408.5882v2.pdf)]
```python
def train_cnn_lstm(x, y, wv_matrix):
    tf.keras.backend.clear_session()
    
    model = Sequential()
    model.add(Embedding(wv_matrix.shape[0], wv_matrix.shape[1], mask_zero=False, 
                        weights=[wv_matrix], input_length=config["MAX_LEN"], trainable=False))
    model.add(Dropout(config['dropout_rate']))
    model.add(Conv1D(filters=config['nb_filter'], kernel_size=config['filter_length'], padding=config['border_mode']))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=config['pool_length']))
    model.add(LSTM(units=config['lstm_cell'], return_sequences=False))
    model.add(Dense(config['fc_cell']*2))
    model.add(Dropout(config['dropout_rate']))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=config['loss'], optimizer=config['optimizer'], metrics=[get_f1])

    print("CNN + LSTM: \n")
    print("="*20, "Start Training", "="*20)

    path = 'weights\{}_weights.hdf5'.format("cnn_lstm")
    model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=3, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    history = model.fit(
        x, y, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=1, 
        callbacks=[model_checkpoint, early_stopping, reduce_lr])
    
    return history, model
```

#### Visualisation
Define two ploting functions to visualise the history of accuracy and loss.

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

    f1 = history.history['get_f1']
    val_f1 = history.history['val_get_f1']

    plt.figure(figsize=(20,10))
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
    plt.ylabel('loss')

    plt.subplot(2,1,2)
    plt.title('F1 Score')
    plt.plot(range(epochs), f1, marker='.', label='acc')
    plt.plot(range(epochs), val_f1, marker='.', label='val_acc')
    plt.legend(loc='best')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.set_facecolor('snow')
    plt.grid(color='lightgray', linestyle='-', linewidth=1)
    plt.xlabel('epoch')
    plt.ylabel('f1')
    plt.show()
```

### Performance
Compare the performance among five deep learning models.
* Simple RNN
![Simple RNN](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/simple_rnn.png)

* GRU
![GRU](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/gru.png)

* LSTM
![LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/lstm.png)

* BiLSTM
![BiLSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/bilstm.png)

* CNN + LSTM
![CNN + LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn_lstm.png)

### Evaluation
In training a neural network, f1 score is an important metric to evaluate the performance of classification models, especially for unbalanced classes where the binary accuracy is useless.

```python
def predict(sentences, model):
    x_test = tokenizer.texts_to_sequences(sentences)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
    y_prob = model.predict_classes(x_test)
    y_pred = y_prob.squeeze()   
    return y_pred
```


```python
# Accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
# Precision tp / (tp + fp)
precision = precision_score(y_test, y_pred)
print('Precision: %f' % precision)
# Recall: tp / (tp + fn)
recall = recall_score(y_test, y_pred)
print('Recall: %f' % recall)
# F1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)
```

||Simple RNN|GRU|LSTM|BiLSTM|CNN + LSTM|
|---|---|---|---|---|---|
|Accuracy|0.622|0.5742|0.6077|0.5981|0.4928|
|F1 Score|0.2178|0.3308|0.2807|0.1064|0.4362|
|Total Params||||||
|Trainable Params||||||
|Non-trainable Params||||||

#### Confusion Matrix
```shell
Simple RNN: 
     [[119  75]
     [  4  11]]

GRU: 
     [[98 64]
     [25 22]]

LSTM:
     [[111  70]
     [ 12  16]]

BiLSTM: 
     [[120  81]
     [  3   5]]

Cnn + LSTM:
     [[62 45]
     [61 41]]
```

## Future Roadmap
It is completely possible to use only raw text as input for making predictions. The most important thing is to extract the relevant features from this raw source of data. Although the models don't perform well and need more improvement, I have done a practise with a full harvest.

Text binary classifier is a meat-and-potatoes issue for most sentiment analysis, and there are still many things can be done on this task. Future works might construct the multi-class text classifier to separate customers' reviews into different issue types. (e.g. Function, UI, Crash, Truncate, Subscription, Server, Enhancement, Other)
