from typing import Any

from selenium import webdriver
import chromedriver_autoinstaller
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_auto_update import check_driver
# Pass in the folder used for storing/downloading chromedriver
import time
import os
import json
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import scipy as sc
import numpy as np
import requests as rs
import re
#function to open the page of authors
Full_list=[]
string = 'https://www.bookdepository.com'
def openSp(file_name):
    x = rs.get(file_name)
    soup=BeautifulSoup(x.content,'html.parser')
    print(x.status_code)
    return soup
def findLinkBlock(soup):#the function to find the list of autors and link to books list
    arr_div=soup('div',attrs={'class':'content-wrap'})
    authors=list()
    links_authors=list()
    for ref in arr_div:
        link=ref.find_all('a',attrs={'class':'link-item-text-only link-item'})
        for lk in link:
            authors.append(lk.get_text().strip())
            links_authors.append(lk.get('href'))
    return pd.DataFrame({'Author':authors,'Author-Link':links_authors})
def count_pages(page_link):#function to count the numbers of pages
    count=1
    list_pages=list()
    count_soup = openSp(page_link)
    link = count_soup.find("li", id="next-top")
    print("start loop")
    while link != None:
        next_page =page_link+"?page="+str(count)
        print(next_page)
        list_pages.append(next_page)
        count_soup = openSp(next_page)
        count += 1
        link = count_soup.find("li", id="next-top")
    print("end loop")
    return list_pages


def seleniumWork(df):#function of selenium to get the info and insert to csv file
    serv = Service(ChromeDriverManager().install())
    book_name={}
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(serv.path)
    links_list=df['Author-Link']
    autors_list=df['Author']
    list_page=string+links_list[0]
    k=0
    for j in range(0,len(links_list)):#loop for get all books of autors
        page_number=1
        linkes_page_list = count_pages(string+links_list[j])
        driver.get(list_page)
        ps = driver.page_source
        soup = BeautifulSoup(ps, 'html.parser')
        for i in range(1,len(linkes_page_list)):#loop to get all pages of html with book list
            page_list_book = soup.find_all('div', attrs={'class': 'book-item'})
            for book in page_list_book:
                try:
                    driver.get(string + book.find('a').get('href'))
                except:
                    break
                ps = driver.page_source
                soup = BeautifulSoup(ps, 'html.parser')
                name = soup.find('h1').get_text()
                try:
                    Num_of_pages = soup.find("span", itemprop="numberOfPages").get_text()
                except:
                    Num_of_pages = None
                try:
                    value_rating = soup.find("span", class_="rating-count").get_text()
                    rating_count = " ".join(value_rating.split())
                except:
                    rating_count = None
                try:
                    value_rating = soup.find("div", itemprop="ratingValue").get_text()
                    rating = " ".join(value_rating.split())
                except:
                    rating = None
                try:
                    price = soup.find("span", class_="sale-price").get_text()
                except:
                    price = None
                try:
                    categories = soup.ol.find_all('a')
                except:
                    break

                the_text = ""
                for cat in categories:#loop to convert the string list of categories to list
                    str = re.search('\w', cat.get_text()).string
                    the_text += str + ','
                res = " ".join(the_text.split())
                res = res.split(',')
                result = [val for val in res if val != ""]#for delet the empty items
                book_name[name] = {"Ganer": list(), "rating": rating, "rating_count": rating_count,
                               "pages": Num_of_pages, "price": price,
                               "Author": autors_list[k]}
                book_name[name]["Ganer"].append(result)
                time.sleep(5)
            print(i)
            try:
                driver.get(linkes_page_list[page_number])
            except:
                break
            page_number += 1
            ps = driver.page_source
            soup = BeautifulSoup(ps, 'html.parser')
            df = pd.DataFrame.from_dict(book_name)
            df.to_csv("material2.csv")#upadet data to csv file
            time.sleep(5)
        k+=1


    driver.quit()#close driver
    print("end of the selenium work finish whith out problems")
    return book_name


def bildAutorsData(book_link, Autor_link):
    soup = openSp(book_link + Autor_link)
    link = soup.find("li", id="next-top")
    while link != None:
        print("here")
        next_page = link.find("a").get('href')
        soup = openSp(book_link + next_page)
        link = soup.find("li", id="next-top")
    return None
soup_result=openSp('https://www.bookdepository.com/popular-authors')
df=findLinkBlock(soup_result)
string='https://www.bookdepository.com'
info=seleniumWork(df)
df=pd.DataFrame.from_dict(info)
df.to_csv("material.csv")
print(df)