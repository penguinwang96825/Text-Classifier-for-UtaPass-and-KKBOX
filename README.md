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
1. Preparing [selenium](https://pypi.org/project/selenium/), [beautiful soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/), and [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html).
```import time
from bs4 import BeautifulSoup
import sys, io
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.proxy import *
import pandas as pd
```
2. Doing text pre-processing after installing [MeCab](https://pypi.org/project/mecab-python-windows/), [neologdn](https://pypi.org/project/neologdn/), [re](https://docs.python.org/3.6/library/re.html), and [emoji](https://pypi.org/project/emoji/)
```
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
```
no_of_reviews = 1000
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

from selenium.common.exceptions import NoSuchElementException        
def check_exists_by_xpath(xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True
reviews = pd.DataFrame(columns = ["review", "Author Name", "Review Date", "Review Ratings", 
                                  "Review Body", "Developer Reply"])
temp = {"review": 0, "Author Name": "", "Review Date": "", "Review Ratings": 0, 
        "Review Body": "", "Developer Reply": ""}

def replace_value_with_definition(key_to_find, definition):
    for key in temp.keys():
        if key == key_to_find:
            temp[key] = definition
```
4. Start crawling the web (Reference: https://github.com/ranjeet867/google-play-crawler)
```
driver = webdriver.Chrome(r"./chromedriver")
wait = WebDriverWait(driver, 10)

# Append your app store urls here
urls = ["https://play.google.com/store/apps/details?id=com.kddi.android.UtaPass&hl=ja"]

for url in urls:

    driver.get(url)

    page = driver.page_source

    soup_expatistan = BeautifulSoup(page, "html.parser")

    expatistan_table = soup_expatistan.find("h1", class_="AHFaub")

    print("App name: ", expatistan_table.string)

    expatistan_table = soup_expatistan.findAll("span", class_="htlgb")[4]

    print("Installs Range: ", expatistan_table.string)

    expatistan_table = soup_expatistan.find("meta", itemprop="ratingValue")

    print("Rating Value: ", expatistan_table['content'])

    expatistan_table = soup_expatistan.find("meta", itemprop="reviewCount")

    print("Reviews Count: ", expatistan_table['content'])

    soup_histogram = soup_expatistan.find("div", class_="VEF2C")

    rating_bars = soup_histogram.find_all('div', class_="mMF0fd")

    for rating_bar in rating_bars:
        print("Rating: ", rating_bar.find("span").text)
        print("Rating count: ", rating_bar.find("span", class_="L2o20d").get('title'))

    # open all reviews
    url = url + '&showAllReviews=true'
    driver.get(url)
    time.sleep(5) # wait dom ready
    for i in range(1,25):
        try:
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);') # scroll to load other reviews
            time.sleep(2)
            if check_exists_by_xpath('//*[@id="fcxH9b"]/div[4]/c-wiz[2]/div/div[2]/div/div[1]/div/div/div[1]/div[2]/div[2]/div/content/span'):
                driver.find_element_by_xpath('//*[@id="fcxH9b"]/div[4]/c-wiz[2]/div/div[2]/div/div[1]/div/div/div[1]/div[2]/div[2]/div/content/span').click()
                time.sleep(2)
        except:
            pass

    page = driver.page_source

    soup_expatistan = BeautifulSoup(page, "html.parser")
    expand_pages = soup_expatistan.findAll("div", class_="d15Mdf")
    counter = 1
    items = []
    
    for expand_page in expand_pages:
        try:
            # print("\n===========\n")

            review = str(counter)
            
            Author_Name = str(expand_page.find("span", class_="X43Kjb").text)
            
            Review_Date = str(expand_page.find("span", class_="p2TkOb").text)
            
            reviewer_ratings = expand_page.find("div", class_="pf5lIe").find_next()['aria-label'];
            reviewer_ratings = reviewer_ratings.split('/')[0]
            reviewer_ratings = ''.join(x for x in reviewer_ratings if x.isdigit())
            Reviewer_Ratings = int(reviewer_ratings)

            Review_Body = str(expand_page.find("div", class_="UD7Dzf").text)
            Review_Body_cleaned = clean_text(Review_Body)
            Review_Body_string = ''.join(Review_Body_cleaned)
            
            developer_reply = expand_page.find_parent().find("div", class_="LVQB0b")
            if hasattr(developer_reply, "text"):
                Developer_Reply = str(developer_reply.text)
            else:
                Developer_Reply = ""
            
            counter += 1           
            item = {
                    "review": counter - 1,
                    "Author Name": Author_Name,
                    "Reviewer Ratings": Reviewer_Ratings,
                    "Review Date": Review_Date,
                    "Review Body": Review_Body_string,
                    "Developer Reply": Developer_Reply
                    }
            items.append(item)
                
        except:
            pass
driver.quit()
```
5. Transforming the data into dataframe using pandas, and removing the rows which contain empty cell.
```
df = pd.DataFrame(items, columns = ["review", "Author Name", "Review Date", "Reviewer Ratings", 
                                    "Review Body", "Developer Reply"])
                                    import numpy as np
df['Review Body'].replace('', np.nan, inplace=True)
df.dropna(subset=['Review Body'], inplace=True)
```
![GitHub Logo](https://github.com/penguinwang96825/Text_Classification/blob/master/image/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7%202019-05-08%20%E4%B8%8B%E5%8D%883.38.26.png)
