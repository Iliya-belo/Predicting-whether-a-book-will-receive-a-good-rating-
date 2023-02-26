#import DataFromWebSite
import pandas as pd
def open_scv(file_name):#function to open csv file
    df=pd.read_csv(file_name)
    return  df
df1=open_scv('material.csv')
df2=open_scv('material2.csv')
result =pd.concat([df1.T,df2.T])#change the data from row to be colmn
result.to_csv('book_data.csv')