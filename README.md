# Text_Classification
Text classification for UtaPass and KKBOX total reviews using different machine learning models.

## Introduction
This analysis is based on text data of UtaPass and KKBOX reviews on Google Play platform. As a KKStreamer from KKBOX, I have always wanted to analyze and classifify the polarity on app reviews. Concretely, I crawled the data using web crawler technique, which is an Internet bot that systematically browses the World Wide Web, and further using different deep learning models (Simple RNN, LSTM, Bi-directional LSTM, GRU, and CNN_LSTM).

## Data Source
1. [UtaPass reviews on Google Play](https://play.google.com/store/apps/details?id=com.kddi.android.UtaPass&hl=ja&showAllReviews=true)
2. [KKBOX reviews on Google Play](https://play.google.com/store/apps/details?id=com.skysoft.kkbox.android&hl=ja&showAllReviews=true)

## Research Questions and Bottleneck
* Do every reviews have sentiment words or charateristic of polarity?
* Do text pre-processing (remove stop words, remove punctuation, remove bad characters) be neccessary? 
* Is there any useless , redundant or even invalid information about the reviews? Do we need to utilize the method such as Anomaly Detection?
* Is there any online user make fake comments to affect other online users on the net? 
  > Luca shows that when a product or business has increased the +1 star rating, it increases revenue by 5-9%. Due to the financial benefits associated with online reviews, paid or prejudiced reviewers write fake reviews to mislead a product or business. [[M.Luca, “Reviews, reputation, and revenue: The case of yelp.com,” Harvard Business School
Working Papers, 2011](https://www.hbs.edu/faculty/Publication%20Files/12-016_a7e4a5a2-03f9-490d-b093-8f951238dba2.pdf)]

## Preparation
1. Preparing selenium, beautiful soup, and pandas.
```import time
from bs4 import BeautifulSoup
import sys, io
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.proxy import *
import pandas as pd```
