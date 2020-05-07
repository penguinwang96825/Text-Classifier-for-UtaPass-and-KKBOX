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

4. [Janome](https://mocobeta.github.io/janome/en/) is written in pure Python including the built-in dictionary and the language model.
* Run `pip install janome` in terminal.

5. BERT for Japanese from Huggingface (update: 2020/04/21)
* Reference: Nekoumei's Qiita [[link](https://qiita.com/nekoumei/items/7b911c61324f16c43e7e)]
* Reference: Hottolink, Inc. pretrained BERT model [[link](https://github.com/hottolink/hottoSNS-bert)]
* cl-tohoku's [GitHub](https://github.com/cl-tohoku/bert-japanese)

## Main Code for Crawling

### Scroll-down Feature and Click-button Feature
```python
def check_exists_by_xpath(xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True

def scrollDownPage():
    # Xpath of "もっと見る" bottom
    button = '//*[@id="fcxH9b"]/div[4]/c-wiz/div/div[2]/div/div[1]/div/div/div[1]/div[2]/div[2]/div/span/span'
    
    # Keep scrolling down until to the very bottom
    keep_scrolling = True
    while keep_scrolling:
        try: 
            # Scroll down to the bottom
            for _ in range(5):
                try: 
                    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                    time.sleep(1 + random.random())
                except:
                    break
            # Click "もっと見る"
            if check_exists_by_xpath(button):
                driver.find_element_by_xpath(button).click()
                time.sleep(2 + random.random())
            else:
                # Stop scrolling down
                keep_scrolling = False
        except: 
            pass
```

### Start Crawling
```python
def open_google_play_reviews(url):
    driver_path = r"C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\chromedriver.exe"
    driver = webdriver.Chrome(driver_path)
    time.sleep(2 + random.random())
    driver.get(url)
    time.sleep(5 + random.random())
    
    scrollDownPage()
    soup = BeautifulSoup(driver.page_source, "html.parser")
    time.sleep(3 + random.random())
    driver.quit()
    return soup
    
def convert_soup_to_dataframe(soup):
    reviews = soup.find(name="div", attrs={"jsname": "fk8dgd"})
    reviews_list = reviews.find_all(name="div", attrs={"jscontroller": "H6eOGe"})
    reviews_all = []
    for i in range(len(reviews_list)):
        name = reviews_list[i].find(name="span", attrs={"class": "X43Kjb"}).string
        date = reviews_list[i].find(name="span", attrs={"class": "p2TkOb"}).string
        rating = reviews_list[i].find(name="div", attrs={"class": "pf5lIe"}).find(name="div").get("aria-label")
        rating = int(rating.split("/")[0][-1])
        content = reviews_list[i].find(name="span", attrs={"jsname": "bN97Pc"}).string
        like = reviews_list[i].find(name="div", attrs={"class": "jUL89d y92BAb"}).string
        reviews_all.append([name, date, rating, content, like])
    df = pd.DataFrame(reviews_all)
    df.columns = ["name", "date", "rating", "content", "like"]
    return df
    
def crawl(url):
    print("Parsing soup from url...")
    soup = open_google_play_reviews(url)
    print("Done parsing soup from url.")
    df = convert_soup_to_dataframe(soup)
    return df
```

### Data Storage
There are 12498 reviews in total. 

```python
df_all.to_csv("all_20200423.csv")
```

Take a look at the dataframe.
||Author Name|Review Date|Reviewer Ratings|Review Content|
|---|---|---|---|---|
|195|眞也大平|2018年12月4日|1|聴い途中止まる強制終了する止まる辞め|
|13|狼音牙|2019年4月22日|1|LISMO使えなっ早くもどせうたパスLISMO使い|
|47|美能孝行|2019年3月12日|3|アルバム曲名読み方登録それ名前反映不具合かなり継続技術改善でき諦め使いあり|
|142|梅川洋子|2019年2月14日|4|いつ聴けるいい|
|45|わんたった|2019年4月27日|1|アンストアプリ残っ|

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

#### Import Packages
```python
import warnings
import re
import emoji
import MeCab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import tensorflow as tf
import keras.backend as K
from keras import regularizers
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
from gensim.models.word2vec import Word2Vec
```

#### Load Data In
First, split dataframe into two categories: positive and negative. Second, do some text preprocessing. For instance, if rating is lower than 3 stars, label it as negative.

```python
df = pd.read_csv(r"C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\data\all_20200423.csv")
# create the label
df["label"] = df["rating"].apply(lambda x: 0 if int(x) <= 3 else 1)
# select only relevant columns
df = df[["content", "label"]]
df["content"] = df["content"].map(str)
df.head(5)
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
 80.14077         55        84
 85.21870         68        46
 90.21044         86        36
 92.06349         96        21
 95.07290         118       14
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
    "nb_epoch": 1000, 
    "validation_split": 0.20, 
    "shuffle": True
}
```

#### Data Pre-processing
1. Remove emoji.
2. Remove punctuation
3. Remove digits.
4. Tokenise sentence using MeCab.

```python
def create_mecab_list(text):
    pos_list = [10, 11, 31, 32, 34]
    pos_list.extend(list(range(36,50)))
    pos_list.extend([59, 60, 62, 67])

    mecab_list = []
    mecab = MeCab.Tagger("-Owakati")
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
    allchars = [string for string in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    cleaned_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return cleaned_text

def clean_text(text):
    # Remove emoji
    text = give_emoji_free_text(text)
    # Remove punctuation
    text = re.sub(r'[^\w\d\s]+', '', text)
    # Remove digits
    text = ''.join([i for i in text if not i.isdigit()]) 
    # Tokenize the sentence
    tokenised_text_list = create_mecab_list(text)
    return tokenised_text_list
```

#### Create Word Index
```python
def create_word2index_and_index2word(df):
    df["cleaned_text"] = df["content"].apply(clean_text)
    sum_list = []
    for index, row in df.iterrows():
        sum_list += row["cleaned_text"]

    word2index = dict()
    index2word = dict()
    num_words = 0
    for word in sum_list:
        if word not in word2index:
            # First entry of word into vocabulary
            word2index[word] = num_words
            index2word[num_words] = word
            num_words += 1
    
    return word2index, index2word

def convert_tokens_to_ids(tokens_list, word2index):
    ids_list = []
    for token in tokens_list:
        if word2index.get(token, None) != None:
            ids_list.append(word2index[token])
    return ids_list

def remove_empty_ids_rows(df):
    empty = (df['ids'].map(len) == 0)
    return df[~empty]

word2index, index2word = create_word2index_and_index2word(df)
df["ids"] = df["content"].apply(lambda x: convert_tokens_to_ids(clean_text(x), word2index))
df = remove_empty_ids_rows(df)
df.head()
```

||content|label|length|cleaned_text|ids|
|---|---|---|---|---|---|
|0   |アプリをダウンロードしたばかりで、バックグラウンドで聴いています。星の理由は曲のダウンロード...   |0   |211 |[アプリ, ダウンロード, バックグラウンド, 聴い, 理由, ダウンロード, 出来, 聴い...   |[0, 1, 2, 3, 4, 1, 5, 3, 6, 7, 8, 9, 10, 11, 1...|
|1   |nan |1   |3   |[nan]   |[26]|
|2   |ダウンロードはネットが必要ですが、その後はオフラインで聞くことが出来てとても便利です。 オフ...   |1   |172 |[ダウンロード, ネット, 必要, その後, オフライン, 聞く, 出来, 便利, オフライ...   |[1, 27, 28, 29, 30, 31, 5, 32, 30, 33, 34, 35,...|
|3   |広告をあまり見たくない方は、下のタブにある本人→右上のアイコンを押すと、30秒間の広告を見る...   |1   |124 |[広告, タブ, ある, 本人, アイコ, 押す, 広告, 見る, 代わり, 時間, 広告,...   |[14, 42, 43, 44, 45, 46, 14, 47, 48, 49, 14, 5...|
|4   |音楽をダウンロードしようと思ったら、ダウンロードマークが無くて、追加しかない状態だった。その...   |0   |121 |[音楽, ダウンロード, しよ, 思っ, ダウンロード, マーク, 無く, 追加, ない, ...   |[15, 1, 62, 63, 1, 64, 65, 35, 66, 67, 68, 69,...|

#### Split Data
Split the data into training data (80%) and testing data (20%).
* Training set: a subset to train a model
* Testing set: a subset to test the trained model

```python
x = df["ids"].map(lambda x: np.array(x))
x = sequence.pad_sequences(x, maxlen=config["MAX_LEN"], padding="post")
print("Features: \n", x)
y = df["label"].values
print("Labels: \n", y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=17)
```

#### Word Embedding
Build the word2vec model to do word embedding. [[Reference](https://github.com/philipperemy/japanese-words-to-vectors/blob/master/README.md)]

Training a Japanese Wikipedia Word2Vec Model by Gensim and Mecab: 
 * Kyubyong Park's [GitHub](https://github.com/Kyubyong/wordvectors)
 * Omuram's [Qiita](https://qiita.com/omuram/items/6570973c090c6f0cb060)
 * TextMiner's [Website](https://textminingonline.com/training-a-japanese-wikipedia-word2vec-model-by-gensim-and-mecab)
```python
def get_embedding_index(model_path):
    w2v = Word2Vec.load(model_path)
    embedding_index = {}
    for word in w2v.wv.vocab:
        embedding_index[word] = w2v.wv.word_vec(word)
    print('Loaded {} word vectors.'.format(len(embedding_index)))

    return embedding_index

def get_embedding_matrix(word2index, embeddings_index, embed_size):
    embedding_matrix = np.zeros((len(word2index) + 1, embed_size))
    for word, i in word2index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words found in embedding index will be pretrained vectors.
            embedding_matrix[i+1] = embedding_vector
        else:
            # words not found in embedding index will be random vectors with certain mean&std.
            embedding_matrix[i+1] = np.random.normal(0.053, 0.3146, size=(1, embed_size))[0]

    # save embedding matrix
    # embed_df = pd.DataFrame(embedding_matrix)
    # embed_df.to_csv(self.path_embedding_matrix, header=None, sep=' ')

    return embedding_matrix

embedding_index = get_embedding_index(
    r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\word2vec\ja.bin')
embedding_matrix = get_embedding_matrix(word2index, embedding_index, embed_size=300)
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
def train_simple_rnn(x_train, y_train, wv_matrix):
    tf.keras.backend.clear_session()
    
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = SimpleRNN(units=config["output_size"], activation=config["rnn_activation"])(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("="*20, "Start Training Simple RNN", "="*20)

    path = 'weights\{}_weights.hdf5'.format("rnn")
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        x_train, y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model
```

##### GRU
```python
def train_gru(x_train, y_train, wv_matrix):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = GRU(units=config["output_size"], return_sequences=False)(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("="*20, "Start Training GRU", "="*20)

    path = 'weights\{}_weights.hdf5'.format("gru")
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        x_train, y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model
```

##### LSTM
```python
def train_lstm(x_train, y_train, wv_matrix):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = LSTM(units=config['lstm_cell'], return_sequences=False)(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("="*20, "Start Training LSTM", "="*20)

    path = 'weights\{}_weights.hdf5'.format("lstm")
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        x_train, y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model
```

##### BiLSTM
```python
def train_bilstm(x_train, y_train, wv_matrix):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = Bidirectional(LSTM(units=config['output_size'], return_sequences=True))(x)
    x = Bidirectional(LSTM(units=config['output_size'], return_sequences=False))(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(config['fc_cell'])(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("="*20, "Start Training BiLSTM", "="*20)

    path = 'weights\weights.hdf5'
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        x_train, y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model
```

##### CNN-LSTM
```python
def train_cnn_lstm(x_train, y_train, wv_matrix):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = Conv1D(filters=config['nb_filter'], kernel_size=config['filter_length'], padding=config['border_mode'])(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=config['pool_length'])(x)
    x = LSTM(units=config['lstm_cell'], return_sequences=False)(x)
    x = Dense(config['fc_cell']*2)(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    model.compile(loss=config['loss'], optimizer=config['optimizer'], metrics=[get_f1])

    print("="*20, "Start Training CNN-LSTM", "="*20)

    path = 'weights\{}_weights.hdf5'.format("cnn_lstm")
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        x_train, y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model
```

##### CNN-static
Based on "Convolutional Neural Networks for Sentence Classification" written by Yoon Kim [[paper link](http://arxiv.org/pdf/1408.5882v2.pdf)]
```python
# Yoon Kim's paper: Convolutional Neural Networks for Sentence Classification
# Reference from https://www.aclweb.org/anthology/D14-1181.pdf
def train_cnn_static(x_train, y_train, wv_matrix, trainable=False):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=trainable)(x_input)
    
    x_conv_1 = Conv1D(filters=config['nb_filter'], 
                      kernel_size=3, 
                      padding=config['border_mode'], 
                      kernel_regularizer=regularizers.l2(3))(x)
    x_conv_1 = BatchNormalization()(x_conv_1)
    x_conv_1 = Activation("relu")(x_conv_1)
    x_conv_1 = MaxPooling1D(pool_size=(config["MAX_LEN"]-3+1), strides=1, padding="valid")(x_conv_1)
    
    x_conv_2 = Conv1D(filters=config['nb_filter'], 
                      kernel_size=4, 
                      padding=config['border_mode'], 
                      kernel_regularizer=regularizers.l2(3))(x)
    x_conv_2 = BatchNormalization()(x_conv_2)
    x_conv_2 = Activation("relu")(x_conv_2)
    x_conv_2 = MaxPooling1D(pool_size=(config["MAX_LEN"]-4+1), strides=1, padding="valid")(x_conv_2)
    
    x_conv_3 = Conv1D(filters=config['nb_filter'], 
                      kernel_size=5, 
                      padding=config['border_mode'], 
                      kernel_regularizer=regularizers.l2(3))(x)
    x_conv_3 = BatchNormalization()(x_conv_3)
    x_conv_3 = Activation("relu")(x_conv_3)
    x_conv_3 = MaxPooling1D(pool_size=(config["MAX_LEN"]-5+1), strides=1, padding="valid")(x_conv_3)
    
    main = Concatenate(axis=1)([x_conv_1, x_conv_2, x_conv_3])
    main = Flatten()(main)
    main = Dropout(config['dropout_rate'])(main)
    main = Dense(units=1)(main)
    main = Activation('sigmoid')(main)
    model = Model(inputs=x_input, outputs=main)

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("="*20, "Start Training CNN-static", "="*20)

    path = 'weights\{}_weights.hdf5'.format("lstm")
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        x_train, y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model
```

##### CNN-multichannel
Based on "Convolutional Neural Networks for Sentence Classification" written by Yoon Kim [[paper link](http://arxiv.org/pdf/1408.5882v2.pdf)]
```python
def train_cnn_multichannel(x_train, y_train, wv_matrix, trainable=False):
    tf.keras.backend.clear_session()
    
    # Channel 1
    x_input_1 = Input(shape=(config["MAX_LEN"], ))
    embedding_1 = Embedding(
        wv_matrix.shape[0], 
        wv_matrix.shape[1], 
        weights=[wv_matrix], 
        trainable=trainable)(x_input_1)
    x_conv_1 = Conv1D(
        filters=config['nb_filter'], 
        kernel_size=3, 
        padding=config['border_mode'], 
        kernel_regularizer=regularizers.l2(3))(embedding_1)
    x_conv_1 = BatchNormalization()(x_conv_1)
    x_conv_1 = Activation("relu")(x_conv_1)
    x_conv_1 = MaxPooling1D(pool_size=(config["MAX_LEN"]-3+1), strides=1, padding="valid")(x_conv_1)
    flat_1 = Flatten()(x_conv_1)
    
    # Channel 2
    x_input_2 = Input(shape=(config["MAX_LEN"], ))
    embedding_2 = Embedding(
        wv_matrix.shape[0], 
        wv_matrix.shape[1], 
        weights=[wv_matrix], 
        trainable=trainable)(x_input_2)
    x_conv_2 = Conv1D(
        filters=config['nb_filter'], 
        kernel_size=3, 
        padding=config['border_mode'], 
        kernel_regularizer=regularizers.l2(3))(embedding_2)
    x_conv_2 = BatchNormalization()(x_conv_2)
    x_conv_2 = Activation("relu")(x_conv_2)
    x_conv_2 = MaxPooling1D(pool_size=(config["MAX_LEN"]-3+1), strides=1, padding="valid")(x_conv_2)
    flat_2 = Flatten()(x_conv_2)
    
    # Channel 1
    x_input_3 = Input(shape=(config["MAX_LEN"], ))
    embedding_3 = Embedding(
        wv_matrix.shape[0], 
        wv_matrix.shape[1], 
        weights=[wv_matrix], 
        trainable=trainable)(x_input_3)
    x_conv_3 = Conv1D(
        filters=config['nb_filter'], 
        kernel_size=3, 
        padding=config['border_mode'], 
        kernel_regularizer=regularizers.l2(3))(embedding_3)
    x_conv_3 = BatchNormalization()(x_conv_3)
    x_conv_3 = Activation("relu")(x_conv_3)
    x_conv_3 = MaxPooling1D(pool_size=(config["MAX_LEN"]-3+1), strides=1, padding="valid")(x_conv_3)
    flat_3 = Flatten()(x_conv_3)
    
    main = Concatenate(axis=1)([flat_1, flat_2, flat_3])
    main = Dense(units=100)(main)
    main = Activation('relu')(main)
    main = Dropout(config['dropout_rate'])(main)
    main = Dense(units=1)(main)
    main = Activation('sigmoid')(main)
    model = Model(inputs=[x_input_1, x_input_2, x_input_3], outputs=main)

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("="*20, "Start Training CNN-multichannel", "="*20)

    path = 'weights\{}_weights.hdf5'.format("lstm")
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        [x_train, x_train, x_train], y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model
```

##### Text-ResNet
```python
def identity_resnet_block(x, nb_filter):
    x_shortcut = x

    # First component of main path
    res_x = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)

    # Second component of main path
    res_x = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(res_x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)

    # Third component of main path
    res_x = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(res_x)
    res_x = BatchNormalization()(res_x)

    # Final Step: add shortcut value to the main path
    x = Add()([x_shortcut, res_x])
    output = Activation('relu')(x)
    return output

def convolutional_resnet_block(x, nb_filter):
    x_shortcut = x

    # First component of main path
    res_x = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)

    # Second component of main path
    res_x = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(res_x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)

    # Third component of main path
    res_x = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(res_x)
    res_x = BatchNormalization()(res_x)

    # Shortcut path
    x_shortcut = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(x_shortcut)
    x_shortcut = BatchNormalization()(x_shortcut)

    # Final Step: add shortcut value to the main path
    x = Add()([x_shortcut, res_x])
    output = Activation('relu')(x)
    return output

def train_text_resnet(x_train, y_train, wv_matrix):
    tf.keras.backend.clear_session()
    # Resnet for reviews of UtaPass and KKBOX
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(0.25)(x)

    # Stage 1
    x = Conv1D(filters=64, kernel_size=3, strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    # Stage 2
    x = convolutional_resnet_block(x, 64)
    x = identity_resnet_block(x, 64)
    x = identity_resnet_block(x, 64)
    # Stage 3
    x = convolutional_resnet_block(x, 128)
    x = identity_resnet_block(x, 128)
    x = identity_resnet_block(x, 128)
    x = identity_resnet_block(x, 128)
    # Stage 4
    x = convolutional_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    # Stage 5
    x = convolutional_resnet_block(x, 512)
    x = identity_resnet_block(x, 512)
    x = identity_resnet_block(x, 512)
    # Average pool
    x = AveragePooling1D(pool_size = 1)(x)
    # Output layer
    x = Flatten()(x)

    x = Dense(units=1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=1024, activation='relu')(x)
    x = Dense(units=1)(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=x_input, outputs=x)
    model.compile(loss=config["loss"],
                  optimizer=config["optimizer"],
                  metrics=[get_f1])
    
    print("="*20, "Start Training Text-ResNet", "="*20)
    
    path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\notebook\weights\text_resnet_weights.hdf5'
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=50, min_lr=0.001)
    
    history = model.fit(
        x_train, y_train, 
        batch_size=config["batch_size"], 
        epochs=config["nb_epoch"], 
        validation_split=config["validation_split"], 
        shuffle=config["shuffle"], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model
```

#### Start Training

##### Set Optimisers

 - Customised Optimiser: Cocob (Continuous Coin Betting algorithm)
Reference from https://medium.com/@mlguy/adding-custom-loss-and-optimizer-in-keras-e255764e1b7d
```python
class COCOB(Optimizer):
    """Coin Betting Optimizer from the paper:
        https://arxiv.org/pdf/1705.07795.pdf
    """
    def __init__(self, alpha=100, **kwargs):
        """
        Initialize COCOB Optimizer
        Args:
            alpha: Refer to paper.
        """
        super(COCOB, self).__init__(**kwargs)
        self._alpha = alpha
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
    def get_updates(self, params, loss, contraints=None):
        self.updates = [K.update_add(self.iterations, 1)]
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        L = [K.variable(np.full(fill_value=1e-8, shape=shape)) for shape in shapes]
        reward = [K.zeros(shape) for shape in shapes]
        tilde_w = [K.zeros(shape) for shape in shapes]
        gradients_sum = [K.zeros(shape) for shape in shapes]
        gradients_norm_sum = [K.zeros(shape) for shape in shapes]
        for p, g, li, ri, twi, gsi, gns in zip(params, grads, L, reward,
                                                     tilde_w,gradients_sum,
                                                       gradients_norm_sum):
            grad_sum_update = gsi + g
            grad_norm_sum_update = gns + K.abs(g)
            l_update = K.maximum(li, K.abs(g))
            reward_update = K.maximum(ri - g * twi, 0)
            new_w = - grad_sum_update / (l_update * (K.maximum(grad_norm_sum_update + l_update, self._alpha * l_update))) * (reward_update + l_update)
            param_update = p - twi + new_w
            tilde_w_update = new_w
            self.updates.append(K.update(gsi, grad_sum_update))
            self.updates.append(K.update(gns, grad_norm_sum_update))
            self.updates.append(K.update(li, l_update))
            self.updates.append(K.update(ri, reward_update))
            self.updates.append(K.update(p, param_update))
            self.updates.append(K.update(twi, tilde_w_update))
        return self.updates
    def get_config(self):
        config = {'alpha': float(K.get_value(self._alpha)) }
        base_config = super(COCOB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
```

 - Setting of other optimisers.
```python
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
adagrad = Adagrad(learning_rate=0.01)
adadelta = Adadelta(learning_rate=1.0, rho=0.95)
rmsprop = RMSprop(learning_rate=0.001, rho=0.9)
nadam = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
cocob = COCOB()
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

#### Simple RNN

* Simple RNN with Adam
![Simple RNN](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/simple-rnn/rnn-adam.png)

* Simple RNN with Adagrad
![Simple RNN](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/simple-rnn/rnn-adagrad.png)

* Simple RNN with Adadelta
![Simple RNN](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/simple-rnn/rnn-adadelta.png)

* Simple RNN with Nadam
![Simple RNN](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/simple-rnn/rnn-nadam.png)

* Simple RNN with RMSprop
![Simple RNN](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/simple-rnn/rnn-rmsprop.png)

* Simple RNN with SGD
![Simple RNN](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/simple-rnn/rnn-sgd.png)

* Simple RNN with COCOB
![Simple RNN](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/simple-rnn/rnn-cocob.png)

||Adam|SGD|Adagrad|Adadelta|RMSprop|Nadam|COCOB|
|---|---|---|---|---|---|---|---|
|Accuracy|0.7125|0.7283|0.7252|0.7392|0.7260|0.7140|0.7341|
|F1 Score|0.7570|0.7651|0.7627|0.7777|0.7608|0.7501|0.7643|

```console
Training model using adam.
==================== Start Training Simple RNN ====================
Confusion Matrix: 
 [[ 684  343]
 [ 400 1157]]

Training model using sgd.
==================== Start Training Simple RNN ====================
Confusion Matrix: 
 [[ 739  357]
 [ 345 1143]]

Training model using adagrad.
==================== Start Training Simple RNN ====================
Confusion Matrix: 
 [[ 733  359]
 [ 351 1141]]

Training model using adadelta.
==================== Start Training Simple RNN ====================
Confusion Matrix: 
 [[ 731  321]
 [ 353 1179]]

Training model using rmsprop.
==================== Start Training Simple RNN ====================
Confusion Matrix: 
 [[ 750  374]
 [ 334 1126]]

Training model using nadam.
==================== Start Training Simple RNN ====================
Confusion Matrix: 
 [[ 736  391]
 [ 348 1109]]

Training model using cocob.
==================== Start Training Simple RNN ====================
Confusion Matrix: 
 [[ 783  386]
 [ 301 1114]]
```

#### GRU

* GRU with Adam
![GRU](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/gru/gru-adam.png)

* GRU with Adagrad
![GRU](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/gru/gru-adagrad.png)

* GRU with Adadelta
![GRU](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/gru/gru-adadelta.png)

* GRU with Nadam
![GRU](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/gru/gru-nadam.png)

* GRU with RMSprop
![GRU](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/gru/gru-rmsprop.png)

* GRU with SGD
![GRU](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/gru/gru-sgd.png)

* GRU with COCOB
![GRU](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/gru/gru-cocob.png)

||Adam|SGD|Adagrad|Adadelta|RMSprop|Nadam|COCOB|
|---|---|---|---|---|---|---|---|
|Accuracy|0.7755|0.7848|0.7829|0.7748|0.7779|0.7713|0.7763|
|F1|0.8059|0.8114|0.8136|0.8066|0.8104|0.8000|0.8056|

```console
Training model using adam.
==================== Start Training GRU ====================
Confusion Matrix: 
 [[ 800  296]
 [ 284 1204]]

Training model using sgd.
==================== Start Training GRU ====================
Confusion Matrix: 
 [[ 832  304]
 [ 252 1196]]

Training model using adagrad.
==================== Start Training GRU ====================
Confusion Matrix: 
 [[ 799  276]
 [ 285 1224]]

Training model using adadelta.
==================== Start Training GRU ====================
Confusion Matrix: 
 [[ 788  286]
 [ 296 1214]]

Training model using rmsprop.
==================== Start Training GRU ====================
Confusion Matrix: 
 [[ 783  273]
 [ 301 1227]]

Training model using nadam.
==================== Start Training GRU ====================
Confusion Matrix: 
 [[ 811  318]
 [ 273 1182]]

Training model using cocob.
==================== Start Training GRU ====================
Confusion Matrix: 
 [[ 808  302]
 [ 276 1198]]
```

#### LSTM

* LSTM with Adam
![LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/lstm/lstm-adam.png)

* LSTM with Adagrad
![LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/lstm/lstm-adagrad.png)

* LSTM with Adadelta
![LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/lstm/lstm-adadelta.png)

* LSTM with Nadam
![LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/lstm/lstm-nadam.png)

* LSTM with RMSprop
![LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/lstm/lstm-rmsprop.png)

* LSTM with SGD
![LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/lstm/lstm-sgd.png)

* LSTM with COCOB
![LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/lstm/lstm-cocob.png)

||Adam|SGD|Adagrad|Adadelta|RMSprop|Nadam|COCOB|
|---|---|---|---|---|---|---|---|
|Accuracy|0.7628|0.7697|0.7806|0.7833|0.7724|0.7670|0.7786|
|F1 Score|0.7984|0.8030|0.8068|0.8125|0.7968|0.7933|0.8092|

```console
Training model using adam.
==================== Start Training LSTM ====================
Confusion Matrix: 
 [[ 757  286]
 [ 327 1214]]

Training model using sgd.
==================== Start Training LSTM ====================
Confusion Matrix: 
 [[ 776  287]
 [ 308 1213]]

Training model using adagrad.
==================== Start Training LSTM ====================
Confusion Matrix: 
 [[ 833  316]
 [ 251 1184]]

Training model using adadelta.
==================== Start Training LSTM ====================
Confusion Matrix: 
 [[ 811  287]
 [ 273 1213]]

Training model using rmsprop.
==================== Start Training LSTM ====================
Confusion Matrix: 
 [[ 843  347]
 [ 241 1153]]

Training model using nadam.
==================== Start Training LSTM ====================
Confusion Matrix: 
 [[ 827  345]
 [ 257 1155]]

Training model using cocob.
==================== Start Training LSTM ====================
Confusion Matrix: 
 [[ 799  287]
 [ 285 1213]]
```

#### BiLSTM

* BiLSTM with Adam
![BiLSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/bilstm/bilstm-adam.png)

* BiLSTM with Adagrad
![BiLSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/bilstm/bilstm-adagrad.png)

* BiLSTM with Adadelta
![BiLSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/bilstm/bilstm-adadelta.png)

* BiLSTM with Nadam
![BiLSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/bilstm/bilstm-nadam.png)

* BiLSTM with RMSprop
![BiLSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/bilstm/bilstm-rmsprop.png)

* BiLSTM with SGD
![BiLSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/bilstm/bilstm-sgd.png)

* BiLSTM with COCOB
![BiLSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/bilstm/bilstm-cocob.png)

||Adam|SGD|Adagrad|Adadelta|RMSprop|Nadam|COCOB|
|---|---|---|---|---|---|---|---|
|Accuracy|0.7872|0.7740|0.7825|0.7829|0.7864|0.7856|0.7790|
|F1 Score|0.8153|0.8047|0.8122|0.8119|0.8167|0.8151|0.8096|

```console
Training model using adam.
==================== Start Training BiLSTM ====================
Confusion Matrix: 
 [[ 820  286]
 [ 264 1214]]

Training model using sgd.
==================== Start Training BiLSTM ====================
Confusion Matrix: 
 [[ 797  297]
 [ 287 1203]]

Training model using adagrad.
==================== Start Training BiLSTM ====================
Confusion Matrix: 
 [[ 807  285]
 [ 277 1215]]

Training model using adadelta.
==================== Start Training BiLSTM ====================
Confusion Matrix: 
 [[ 812  289]
 [ 272 1211]]

Training model using rmsprop.
==================== Start Training BiLSTM ====================
Confusion Matrix: 
 [[ 802  270]
 [ 282 1230]]

Training model using nadam.
==================== Start Training BiLSTM ====================
Confusion Matrix: 
 [[ 809  279]
 [ 275 1221]]

Training model using cocob.
==================== Start Training BiLSTM ====================
Confusion Matrix: 
 [[ 799  286]
 [ 285 1214]]
```

#### CNN-Static

* CNN-Static with Adam
![CNN-Static](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-static/cnn-static-adam.png)

* CNN-Static with Adagrad
![CNN-Static](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-static/cnn-static-adagrad.png)

* CNN-Static with Adadelta
![CNN-Static](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-static/cnn-static-adadelta.png)

* CNN-Static with Nadam
![CNN-Static](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-static/cnn-static-nadam.png)

* CNN-Static with RMSprop
![CNN-Static](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-static/cnn-static-rmsprop.png)

* CNN-Static with SGD
![CNN-Static](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-static/cnn-static-sgd.png)

* CNN-Static with COCOB
![CNN-Static](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-static/cnn-static-cocob.png)

||Adam|SGD|Adagrad|Adadelta|RMSprop|Nadam|COCOB|
|---|---|---|---|---|---|---|---|
|Accuracy|0.7365|0.7032|0.7101|0.7566|0.7411|0.7051|0.7105|
|F1 Score|07628|0.7850|0.7854|0.7938|0.7802|0.7855|0.7242|

```console
Training model using adam.
==================== Start Training CNN-static ====================
Confusion Matrix: 
 [[ 808  405]
 [ 276 1095]]

Training model using sgd.
==================== Start Training CNN-static ====================
Confusion Matrix: 
 [[ 417  100]
 [ 667 1400]]

Training model using adagrad.
==================== Start Training CNN-static ====================
Confusion Matrix: 
 [[ 464  129]
 [ 620 1371]]

Training model using adadelta.
==================== Start Training CNN-static ====================
Confusion Matrix: 
 [[ 744  289]
 [ 340 1211]]

Training model using rmsprop.
==================== Start Training CNN-static ====================
Confusion Matrix: 
 [[ 728  313]
 [ 356 1187]]

Training model using nadam.
==================== Start Training CNN-static ====================
Confusion Matrix: 
 [[ 427  105]
 [ 657 1395]]

Training model using cocob.
==================== Start Training CNN-static ====================
Confusion Matrix: 
 [[854 518]
 [230 982]]
```

#### CNN-MultiChannel

* CNN-MultiChannel with Adam
![CNN-MultiChannel](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-multichannel/cnn-multichannel-adam.png)

* CNN-MultiChannel with Adagrad
![CNN-MultiChannel](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-multichannel/cnn-multichannel-adagrad.png)

* CNN-MultiChannel with Adadelta
![CNN-MultiChannel](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-multichannel/cnn-multichannel-adadelta.png)

* CNN-MultiChannel with Nadam
![CNN-MultiChannel](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-multichannel/cnn-multichannel-nadam.png)

* CNN-MultiChannel with RMSprop
![CNN-MultiChannel](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-multichannel/cnn-multichannel-rmsprop.png)

* CNN-MultiChannel with SGD
![CNN-MultiChannel](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-multichannel/cnn-multichannel-sgd.png)

* CNN-MultiChannel with COCOB
![CNN-MultiChannel](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-multichannel/cnn-multichannel-cocob.png)

||Adam|SGD|Adagrad|Adadelta|RMSprop|Nadam|COCOB|
|---|---|---|---|---|---|---|---|
|Accuracy|0.7550|0.7477|0.7287|0.7659|0.7438|0.7388|0.7372|
|F1 Score|0.7828|0.7772|0.7410|0.8022|0.7722|0.7700|0.7579|

```console
Training model using adam.
==================== Start Training CNN-multichannel ====================
Confusion Matrix: 
 [[ 811  360]
 [ 273 1140]]

Training model using sgd.
==================== Start Training CNN-multichannel ====================
Confusion Matrix: 
 [[ 795  363]
 [ 289 1137]]

Training model using adagrad.
==================== Start Training CNN-multichannel ====================
Confusion Matrix: 
 [[ 880  497]
 [ 204 1003]]

Training model using adadelta.
==================== Start Training CNN-multichannel ====================
Confusion Matrix: 
 [[ 752  273]
 [ 332 1227]]

Training model using rmsprop.
==================== Start Training CNN-multichannel ====================
Confusion Matrix: 
 [[ 800  378]
 [ 284 1122]]

Training model using nadam.
==================== Start Training CNN-multichannel ====================
Confusion Matrix: 
 [[ 779  370]
 [ 305 1130]]

Training model using cocob.
==================== Start Training CNN-multichannel ====================
Confusion Matrix: 
 [[ 842  437]
 [ 242 1063]]
```

#### CNN-LSTM

* CNN-LSTM with Adam
![CNN-LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-lstm/cnn-lstm-adam.png)

* CNN-LSTM with Adagrad
![CNN-LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-lstm/cnn-lstm-adagrad.png)

* CNN-LSTM with Adadelta
![CNN-LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-lstm/cnn-lstm-adadelta.png)

* CNN-LSTM with Nadam
![CNN-LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-lstm/cnn-lstm-nadam.png)

* CNN-LSTM with RMSprop
![CNN-LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-lstm/cnn-lstm-rmsprop.png)

* CNN-LSTM with SGD
![CNN-LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-lstm/cnn-lstm-sgd.png)

* CNN-LSTM with COCOB
![CNN-LSTM](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/cnn-lstm/cnn-lstm-cocob.png)

||Adam|SGD|Adagrad|Adadelta|RMSprop|Nadam|COCOB|
|---|---|---|---|---|---|---|---|
|Accuracy|0.7790|0.7872|0.7775|0.7752|0.7659|0.7771|0.7705|
|F1 Score|0.8138|0.8115|0.8085|0.8068|0.7948|0.8076|0.7981|

```console
Training model using adam.
==================== Start Training CNN-LSTM ====================
Confusion Matrix: 
 [[ 765  252]
 [ 319 1248]]

Training model using sgd.
==================== Start Training CNN-LSTM ====================
Confusion Matrix: 
 [[ 850  316]
 [ 234 1184]]

Training model using adagrad.
==================== Start Training CNN-LSTM ====================
Confusion Matrix: 
 [[ 795  286]
 [ 289 1214]]

Training model using adadelta.
==================== Start Training CNN-LSTM ====================
Confusion Matrix: 
 [[ 790  287]
 [ 294 1213]]

Training model using rmsprop.
==================== Start Training CNN-LSTM ====================
Confusion Matrix: 
 [[ 807  328]
 [ 277 1172]]

Training model using nadam.
==================== Start Training CNN-LSTM ====================
Confusion Matrix: 
 [[ 799  291]
 [ 285 1209]]

Training model using cocob.
==================== Start Training CNN-LSTM ====================
Confusion Matrix: 
 [[ 819  328]
 [ 265 1172]]
```

#### Text-ResNet

* Text-ResNet with Adam
![Text-ResNet](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/text-resnet/text-resnet-adam.png)

* Text-ResNet with Adagrad
![Text-ResNet](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/text-resnet/text-resnet-adagrad.png)

* Text-ResNet with Adadelta
![Text-ResNet](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/text-resnet/text-resnet-adadelta.png)

* Text-ResNet with Nadam
![Text-ResNet](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/text-resnet/text-resnet-nadam.png)

* Text-ResNet with RMSprop
![Text-ResNet](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/text-resnet/text-resnet-rmsprop.png)

* Text-ResNet with SGD
![Text-ResNet](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/text-resnet/text-resnet-sgd.png)

* Text-ResNet with COCOB
![Text-ResNet](https://github.com/penguinwang96825/Text_Classifier_for_UtaPass_and_KKBOX/blob/master/image/text-resnet/text-resnet-cocob.png)

||Adam|SGD|Adagrad|Adadelta|RMSprop|Nadam|COCOB|
|---|---|---|---|---|---|---|---|
|Accuracy|0.7330|0.7206|0.7450|0.7337|0.7318|0.7260|0.7396|
|F1 Score|0.7667|0.7619|0.7797|0.7679|0.7758|0.7662|0.7839|

```console
Training model using adam.
==================== Start Training Text-ResNet ====================
Confusion Matrix: 
 [[ 760  366]
 [ 324 1134]]

Training model using sgd.
==================== Start Training Text-ResNet ====================
Confusion Matrix: 
 [[ 707  345]
 [ 377 1155]]

Training model using adagrad.
==================== Start Training Text-ResNet ====================
Confusion Matrix: 
 [[ 759  334]
 [ 325 1166]]

Training model using adadelta.
==================== Start Training Text-ResNet ====================
Confusion Matrix: 
 [[ 758  362]
 [ 326 1138]]

Training model using rmsprop.
==================== Start Training Text-ResNet ====================
Confusion Matrix: 
 [[ 692  301]
 [ 392 1199]]

Training model using nadam.
==================== Start Training Text-ResNet ====================
Confusion Matrix: 
 [[ 716  340]
 [ 368 1160]]

Training model using cocob.
==================== Start Training Text-ResNet ====================
Confusion Matrix: 
 [[ 690  279]
 [ 394 1221]]
```

### Evaluation
In training a neural network, f1 score is an important metric to evaluate the performance of classification models, especially for unbalanced classes where the binary accuracy is useless.

```python
def predict(sentences, model):
    y_prob = model.predict(x_test)
    y_prob = y_prob.squeeze()
    y_pred = (y_prob > 0.5) 
    return y_pred
```

#### Accuracy
||Simple-RNN|GRU|LSTM|BiLSTM|CNN-Static|CNN-MultiChannel|CNN-LSTM|Text-ResNet|
|---|---|---|---|---|---|---|---|---|
|Adam|0.7125|0.7755|0.7628|**0.7872**|0.7365|0.7550|0.7790|0.7330|
|SGD|0.7283|**0.7848**|0.7697|0.7740|0.7032|0.7477|**0.7872**|0.7206|
|Adagrad|0.7252|0.7829|0.7806|0.7825|0.7101|0.7287|0.7775|**0.7450**|
|Adadelta|**0.7392**|0.7748|**0.7833**|0.7829|**0.7566**|**0.7659**|0.7752|0.7337|
|RMSprop|0.7260|0.7779|0.7724|0.7864|0.7411|0.7438|0.7659|0.7318|
|Nadam|0.7140|0.7713|0.7670|0.7856|0.7051|0.7388|0.7771|0.7260|
|COCOB|0.7341|0.7763|0.7786|0.7790|0.7105|0.7372|0.7705|0.7396|

#### F1 Score
||Simple-RNN|GRU|LSTM|BiLSTM|CNN-Static|CNN-MultiChannel|CNN-LSTM|Text-ResNet|
|---|---|---|---|---|---|---|---|---|
|Adam|0.7570|0.8059|0.7984|0.8153|07628|0.7828|**0.8138**|0.7667|
|SGD|0.7651|0.8114|0.8030|0.8047|0.7850|0.7772|0.8115|0.7619|
|Adagrad|0.7627|**0.8136**|0.8068|0.8122|0.7854|0.7410|0.8085|0.7797|
|Adadelta|**0.7777**|0.8066|**0.8125**|0.8119|**0.7938**|**0.8022**|0.8068|0.7679|
|RMSprop|0.7608|0.8104|0.7968|**0.8167**|0.7802|0.7722|0.7948|0.7758|
|Nadam|0.7501|0.8000|0.7933|0.8151|0.7855|0.7700|0.8076|0.7662|
|COCOB|0.7643|0.8056|0.8092|0.8096|0.7242|0.7579|0.7981|**0.7839**|

## Future Roadmap
It is completely possible to use only raw text as input for making predictions. The most important thing is to extract the relevant features from this raw source of data. Although the models don't perform well and need more improvement, I have done a practise with a full harvest.

Text binary classifier is a meat-and-potatoes issue for most sentiment analysis, and there are still many things can be done on this task. In future works, I might construct a multi-class text classifier to separate customers' reviews into different issue types. (e.g. Function, UI, Crash, Truncate, Subscription, Server, Enhancement, Other)
