#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import io
import requests




def read_bodyfat():

    """
        Function to read bodyfat data. Target variable is 'Percent body fat from Siri's (1956) equation'
    """
    url="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/bodyfat"
    s=requests.get(url).content
    df=pd.read_csv(io.StringIO(s.decode('utf-8')),sep = ' ',names=[ 'Density determined from underwater weighing',
 'Percent body fat from Siri\'s (1956) equation',
    'Age (years)',
 'Weight (lbs)',
 'Height (inches)',
 'Neck circumference (cm)',
 'Chest circumference (cm)',
 'Abdomen 2 circumference (cm)',
 'Hip circumference (cm)',
 'Thigh circumference (cm)',
 'Knee circumference (cm)',
 'Ankle circumference (cm)',
 'Biceps (extended) circumference (cm)',
'Forearm circumference (cm)',
 'Wrist circumference (cm)]'])
    df.rename(columns = {'Percent body fat from Siri\'s (1956) equation':'target'}, inplace = True)    
    for columns in df.columns[1:]:
        # Changing all columns except first one as they are in format colnum:val to just val
        df[columns] = df[columns].apply(lambda x: x.split(':')[1])
    y=df['target']
    X=df.drop('target',axis=1)
    # Issues with object type, so converting to float type uniformly
    return (X.astype(np.float32),y.astype(np.float32),df.astype(np.float32))





def read_housing():
    """
        Function to read boston housing data. 
    """
    from sklearn.datasets import load_boston
    boston_dataset = load_boston()
    df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    df['target']=boston_dataset.target
    y=df['target']
    X=df.drop('target',axis=1)
    return (X,y,df)





def read_abalone():
    """
        Function to read abalone data. Target variable is 'rings'
    """

    column_names = ["sex", "length", "diameter", "height", "whole weight", 
                "shucked weight", "viscera weight", "shell weight", "rings"]
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", names=column_names)
    df.rename(columns = {'rings':'target'}, inplace = True)
    
    # Dropping sex variable for now. Will think of using other enconding techniques later
    df.drop('sex',axis=1,inplace=True)
    y=df['target']
    X=df.drop('target',axis=1)
    return (X,y,df)



def read_cpusmall():
    """
        Function to read cpusmall data. Target variable is 'usr'
    """

    url="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cpusmall"
    s=requests.get(url).content
    df=pd.read_csv(io.StringIO(s.decode('utf-8')),sep = ' ',names=[ 'lread',
     'lwrite','scall','sread','swrite','fork','exec','rchar','wchar','runqsz','freemem','freeswap','usr'     
    ])
    df.rename(columns = {'usr':'target'}, inplace = True)
    
    for columns in df.columns[1:]:
        # Changing all columns except first one as they are in format colnum:val to just val
        df[columns] = df[columns].apply(lambda x: x.split(':')[1])
    
    y=df['target']
    X=df.drop('target',axis=1)
    return (X.astype(np.float32),y.astype(np.float32),df.astype(np.float32))




def read_cadata():
    """
        Function to read cadata. Target variable is 'median house value'
    """
    url="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata"
    s=requests.get(url).content
    df=pd.read_csv(io.StringIO(s.decode('utf-8')),sep = ' ',names=[ 'median house value', 'median income', 'housing median age', 'total rooms', 'total bedrooms', 'population', 'households', 'latitude', 'longitude'    
    ])
    df.rename(columns = {'median house value':'target'}, inplace = True)
    
    for columns in df.columns[1:]:
        # Changing all columns except first one as they are in format colnum:val to just val
        df[columns] = df[columns].apply(lambda x: x.split(':')[1])
    
    y=df['target']
    X=df.drop('target',axis=1)
    return (X.astype(np.float32),y.astype(np.float32),df.astype(np.float32))







