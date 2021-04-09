#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp 
import os
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from acquire import get_diabetic_data, get_id_data


pd.set_option('display.max_columns', 80)


# In[3]:


def clean_diabetic_data():
    df= get_diabetic_data()
    #dropping unneeded columns
    df=df.drop(columns=['admission_type_id','patient_nbr', 'diag_1','diag_2', 'diag_3', 'readmitted', 'admission_source_id', 'discharge_disposition_id', 'payer_code',
'change', 'medical_specialty', 'num_lab_procedures', 'num_procedures', 'number_emergency','number_outpatient','number_inpatient','number_diagnoses', 'diabetesMed'])
    #replacing ? with nan
    df['weight']=df['weight'].replace('?', np.nan)
    df['race']=df['race'].replace('?', np.nan)
    df['gender']=df['gender'].replace('?', np.nan)
    #dropping nan
    df = df.dropna()
    #making dummy variables
    dummy_df = pd.get_dummies(df[['race', 'gender','age','weight','metformin',
       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']])
    #appending dummy variables to dataframe
    df = pd.concat([df, dummy_df], axis=1)
    #dropping non dummy columns
    df= df.drop(columns=['metformin',
       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone','num_medications','max_glu_serum','A1Cresult'])
    df= df.drop(columns=['repaglinide_No', 'nateglinide_No', 'chlorpropamide_No','glimepiride_No', 'acetohexamide_No','glipizide_No','glyburide_No','tolbutamide_No', 'pioglitazone_No', 'rosiglitazone_No','acarbose_No','miglitol_No', 'troglitazone_No', 'tolazamide_No', 'examide_No', 'citoglipton_No', 'insulin_No','glyburide-metformin_No','glipizide-metformin_No', 'glimepiride-pioglitazone_No','metformin-rosiglitazone_No', 'metformin-pioglitazone_No'])
    return df


# In[ ]:


def data_split(df, stratify_by='time_in_hospital'):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=['time_in_hospital','race', 'gender', 'age', 'weight','insulin'])
    y_train = train['time_in_hospital']
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=['time_in_hospital', 'race', 'gender', 'age', 'weight','insulin'])
    y_validate = validate['time_in_hospital']
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=['time_in_hospital', 'race', 'gender', 'age', 'weight','insulin'])
    y_test = test['time_in_hospital']
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


# In[ ]:


def scaled_data(X_train, X_validate, X_test, y_train, y_validate, y_test):

    # Make the scaler
    scaler = MinMaxScaler()

    # Fit the scaler
    scaler.fit(X_train)

    # Use the scaler
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), columns=X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Make y_values separate dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    #Unscaled data for later
    X_unscaled= pd.DataFrame(scaler.inverse_transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test, X_unscaled


# In[ ]:


def select_kbest(X_train, y_train, k):
    '''
    Takes in the predictors (X_train_scaled), the target (y_train), 
    and the number of features to select (k) 
    and returns the names of the top k selected features based on the SelectKBest class
    '''
    f_selector = SelectKBest(f_regression, k)
    f_selector = f_selector.fit(X_train, y_train)
    X_train_reduced = f_selector.transform(X_train)
    f_support = f_selector.get_support()
    f_feature = X_train.iloc[:,f_support].columns.tolist()
    return f_feature


# In[ ]:


def rfe(X_train, y_train, k):
    '''
    Takes in the predictor (X_train_scaled), the target (y_train), 
    and the number of features to select (k).
    Returns the top k features based on the RFE class.
    '''
    lm = LinearRegression()
    rfe = RFE(lm, k)
    # Transforming data using RFE
    X_rfe = rfe.fit_transform(X_train, y_train)
    #Fitting the data to model
    lm.fit(X_rfe,y_train)
    mask = rfe.support_
    rfe_features = X_train.loc[:,mask].columns.tolist()
    return rfe_features

