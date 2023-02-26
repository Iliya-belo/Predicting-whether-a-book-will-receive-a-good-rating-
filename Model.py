import scipy as sc
import numpy as np
import string
import requests as rs
import re
import os
from IPython.display import display
from statistics import mean
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.stats import *
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
data={'Ganers':['Fiction','Nonfiction','Action and adventure','Art/architecture','Alternate history','Autobiography','Anthology',
                             'Biography','Chick lit','Business','economics','Childrens','Kids','Crafts','hobbies','Classic',
                             'Cookbook','Comic book','Diary','Coming-of-age','Dictionary','Crime','Encyclopedia',
                             'Drama','Guide','Fairytale','Health/fitness','Fantasy','History','Graphic novel','Home','garden',
                             'Historical fiction','Humor','Horror','Journal','Mystery','Math','Paranormal romance','Memoir',
                             'Picture book','Philosophy','Poetry','Prayer','Political thriller','Religion'
                             ,'Romance','Textbook','Satire','True crime','Science fiction','Review','Short story',
                             'Science','Suspense','Self help','Thriller','Sports','leisure','Western','Travel',
                             'Young adult','True crime'],'Count of Books':0}
different_ganer=pd.DataFrame(data=data)
def Classify_ganers(df1,df2):
    place=0
    avg_list_rating=[]
    for mainGaner in df2['Ganers']:
        count=0
        avg=0
        for ganer in df1['Ganer']:
            list=ganer.split(',')
            row_number = df1.index[df1['Ganer'] == ganer].tolist()[0]
            for word in list:
                ad=word.find(mainGaner)
                if ad!=-1:
                    count+=1
                    avg+=df1.loc[row_number,'rating']
        df2.at[place, 'Count of Books']=count
        if count !=0:
            avg_list_rating.append(avg/count)
        else:
            avg_list_rating.append(0)
        place+=1
    sum=0
    for var in df2['Count of Books']:
        sum+=var
    df2['rating of ganer']=avg_list_rating
    return  df2

def bild_pie(ganer_data):
    books_genre_counts = ganer_data.groupby(by=['Ganers'])['rating of ganer'].sum().sort_values(ascending=False)

    labels = ["Genre: %s" % i for i in books_genre_counts.index]

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    fig1.subplots_adjust(0.3, 0, 1, 1)

    _, _ = ax1.pie(books_genre_counts.values, startangle=90)

    ax1.axis('equal')

    plt.legend(
        loc='upper left',
        labels=['%s, %f' % (
            l, v) for l, s, v in zip(labels, books_genre_counts.index, books_genre_counts.values)],
        prop={'size': 11},
        bbox_to_anchor=(0.0, 1),
        bbox_transform=fig1.transFigure
    )

    plt.show()
def add_Main_ganer(df):
    df['Main Ganer']=''
    for data in df['book_name']:
        row_number = df.index[df['book_name'] == data].tolist()[0]
        gan_str=df.loc[row_number,'Ganer']
        list = gan_str.split(',')
        df.at[row_number,'Main Ganer']=list[0]
    return df
def open_scv(file_name):
    df=pd.read_csv(file_name)
    return  df
def graff(df):
    for item in numeric_cols:
        plt.figure()
        plt.hist(df[item], edgecolor='black', histtype='bar')
        plt.title(item)
        plt.yscale('linear')
        plt.xscale('linear')
        plt.show()
def cor(df):
    col_for_corr = ['rating_count', 'rating','pages']
    for item in col_for_corr:
        print(item + ': ' + str(df['price'].corr(df[item], method="spearman")))
        plt.figure()
        plt.scatter(df[item],df['price'])
        plt.title("price and "+item)
        plt.show()
    col_for=['rating_count', 'rating']
    for item in col_for:
        print(item + ': ' + str(df['pages'].corr(df[item], method="spearman")))
        plt.figure()
        plt.scatter(df[item],df['pages'])
        plt.title("pages and "+item)
        plt.show()
    print('rating' + ': ' + str(df['rating_count'].corr(df['rating'], method="spearman")))
    plt.figure()
    plt.scatter(df['rating'], df['rating_count'])
    plt.title("rating and " + 'rating_count')
    plt.show()
def box_plot(df):
    col_for_corr = ['rating_count', 'rating', 'pages','price']
    for name in col_for_corr:
        sns.boxplot(df[name],whis=3)
        plt.show()
def box_coil(df):
    col_for_corr = ['rating_count', 'rating', 'pages', 'price']
    for name in col_for_corr:
        sns.violinplot(df[name])
        plt.show()
def Blocks(df):
    block_list=['rating','rating_count','pages','price']
    for item in block_list:
        plt.figure()
        plt.hist(df[item],edgecolor='black')
        plt.title(item)
        plt.yscale("log")
        plt.show()
def dic_ganers(df):
    count_ganer =[]
    ganer_list=[]
    for data in df['Ganer']:
        data = data.replace("[", "")
        data = data.replace("]", "")
        data = data.replace("'", "")
        info = data.split(',')
        for data in info:
            data=" ".join(data.split())
            ganer_list.append(data)
    for ganer in ganer_list:
        count_ganer.append(ganer_list.count(ganer))
    ganer_list=list(dict.fromkeys(ganer_list))
    ganer_dic=dict(zip(ganer_list,count_ganer))
    return ganer_dic
def graf_func(df):
    m = linear_model.LinearRegression().fit(df.iloc[:, 0:1], df.iloc[:,0:1])
    plt.scatter(x=df['rating'], y=df['price'], c='k', marker='*', label='Digital')
    plt.plot(df['rating'], m.predict(df.iloc[:, 0:1]), 'k', color='blue', linewidth=3)

    plt.xlabel('Digital budget (Thousands of dollars)')
    plt.ylabel('Sales (Thousand units of product)')
    plt.show()
df=open_scv('other_finaly_option1.csv')
cor(df)
Blocks(df)
print(df['pages'].min())
print(df['pages'].max())
sns.histplot(df['pages'])
plt.show()
new_df=add_Main_ganer(df)
Classyfy(new_df)
graff(df)
temp1=Classify_ganers(df,different_ganer)
temp1=temp1[temp1!=0].dropna()
box_plot(df)
box_coil(df)
