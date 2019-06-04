from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.proxy import *
from selenium.webdriver.common.by import By
import time, sys, io
import pandas as pd

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

def openGooglePlayStore():
	wait = WebDriverWait(driver, 10)
	url = "https://play.google.com/store/apps/details?id=com.kddi.android.UtaPass&hl=ja&showAllReviews=true"
	driver.get(url)
	time.sleep(5)

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
