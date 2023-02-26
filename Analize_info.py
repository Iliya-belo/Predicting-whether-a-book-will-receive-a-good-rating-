import scipy as sc
import numpy as np
import string
import requests as rs
import re
import os
from statistics import mean
import pandas as pd
def open_scv(file_name):
    df=pd.read_csv(file_name)
    return  df
def delet_shekel(df):#function to convert the price to be integer
    count=1
    for data in df['price']:
        df.at[count,'price']=data[1:]
        count+=1
    return  df
def delet_str_from_pages(df):#function to convert the string number of pages to be int wih out characters
    count = 1
    for data in df['pages']:
        info = data.split(' ')
        df.at[count, 'pages'] = info[0]
        count += 1
    return df
def delete_count_under(df):#function to delet the item who less 200 of people rating
    df.drop(df[df['rating_count'] < 200].index, inplace=True)
    count=0
    for gan in df['Ganer']:
        if gan=='':
            df.drop(df.index[count],axis=0,inplace=True)
        count+=1
    df.to_csv('other_finaly_option1.csv')
def chage_str_to_num(df):#function to change the string of total rating to be number
    count=1
    for data in df['rating']:
        info=data.split(' ')
        df.at[count,'rating']=info[0]
        count+=1
def info_work(df):
    count=1
    for data in df['rating_count']:
        count_people= ''.join(x for x in data if x.isdigit())
        out = [int(s.replace(' ', '').rstrip(string.ascii_lowercase)) for s in count_people]#bild integers list of numbers
        number=0
        for num in out:#bild the true number
            number=number*10+num
        df.at[count,'rating_count']=number
        count+=1
    df=delet_shekel(df)
    df=delet_str_from_pages(df)

    chage_str_to_num(df)
    drop_dup_in_ganers(df)
    delete_count_under(df)

def drop_dup_in_ganers(df):#function to delete duplicates of ganers
    count=1
    temp_list=[]
    for data in df['Ganer']:

        data=data.replace("[","")
        data=data.replace("]","")
        data = data.replace("'", "")
        data=data.replace("\"","")
        data=data.replace("\'","")
        info=data.split(',')
        for data in info:
            if data not in temp_list:
                temp_list.append(data)
        info.clear()
        for temp in temp_list:
            temp = " ".join(temp.split())
            info.append(temp)
        temp_list.clear()
        ganers=', '.join(info)
        df.at[count, 'Ganer'] =ganers
        count += 1
    return df
df=open_scv('book_after.csv')
print(different_ganer)
df_mun=df.copy()
df_mun=df_mun.dropna()
df_dup=df_mun.rename(columns={'Unnamed: 0':'book_name','0':'Ganer','1':'rating','2':'rating_count','3':'pages','4':'price','5':'Author'})
df_mun.drop(0,axis=0,inplace=True)
print(df_mun)
df_mun=df_mun.drop_duplicates()
df_mun=df_mun.drop(df_mun.columns[[0]],axis = 1)
info_work(df_mun)
