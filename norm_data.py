'''
This module contains a function which splits the data into labels and features, 
performs preprocessing steps and merges it back.
'''

import pandas as pd

from sklearn import preprocessing
from sklearn import decomposition
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import svm
# from sklearn import metrics
# from sklearn import neighbors


def split_data(data):
    '''
    This function splits the dataframe into a dataframe with features and a dataframe with labels.
    '''
    #Create a dataframe with the label
    data_reset = data.reset_index() #Sets the index as the first column
    data_label = data_reset.iloc[:,0:2] #Gets a dataframe with the ID and the label
    df_label = data_label.set_index(data_label.columns[0]) #Sets the ID column as the index

    #Create a dataframe with the features
    data_id = data_reset.iloc[:,0] #Gets a dataframe with the ID
    data_selection = data_reset.iloc[:,2:] #Gets a dataframe with only the features
    return data_selection, data_id, df_label

def merge_data(data_selection, data_id, df_label, data_processed):
    '''
    This function merges the processed data and the labels.
    '''
    #Create a dataframe with the processed data
    df_processed = pd.DataFrame(data_processed) #, columns=data_selection.columns
    df_processed.insert(0, 'ID', data_id.values)  #Inserts a column with the ID
    df_processed = df_processed.set_index(df_processed.columns[0]) #Sets the ID column as the index

    #Merge the label and processed data DataFrames
    data_merged = pd.merge(df_label, df_processed, on='ID')
    return data_merged, df_label, df_processed

def scale_data(data_selection, scaler=preprocessing.StandardScaler()):
    '''
    This function scales the data.
    The default scaler is StandardScaler.

    It returns a dataframe with the merged data and the labels.
    '''
    scaler.fit(data_selection)
    data_scaled = scaler.transform(data_selection)
    return data_scaled

def pca_data(data):
    '''
    This function performs a PCA analysis on the data.
    The number of components is such that the 95% of the variance is retained.
    '''
    pca = decomposition.PCA(n_components=0.95)
    pca.fit(data)
    data_pca = pca.transform(data)

    return data_pca

def rfecv_data(data,labels):
    '''
    This function performs Recursive Feature Elimination with Cross-Validation (RFECV) 
    using a Support Vector Classifier (SVC).
    The input is a dataframe with the features and a dataframe with the labels.
    '''
    svc = svm.SVC(kernel="linear")
    rfecv = feature_selection.RFECV(
        estimator=svc, step=1,
        cv=model_selection.StratifiedKFold(4),
        scoring='roc_auc',
        min_features_to_select=25) #Dit nu een beetje gegokt
    labels = labels.values.ravel() #Makes a 1d array
    rfecv.fit(data, labels) #Fits the rfecv
    selected_features = data.columns[rfecv.support_] #Gets the best number of features
    data_rfecv = data[selected_features] #Gets the best features
    return data_rfecv


def processing_data_scaling(data):
    '''
    This function processes the data.
    First the data is splitted, then processed (scaling) and then merged.
    '''
    data_selection, data_id, df_label = split_data(data) #Splits the data and the label

    data_scaled = scale_data(data_selection, scaler=preprocessing.StandardScaler())
    # data_pca = pca_data(data_scaled)

    data_merged, df_label, df_processed = merge_data(data_selection, data_id, df_label, data_scaled)
    #Merges the processed data and the label
    #data_rfecv = rfecv_data(df_processed,df_label) #Duurt lang dus ff uitzetten
    return data_merged, df_label, df_processed#, data_rfecv

def processing_data_pca(data):
    '''
    This function processes the data.
    First the data is splitted, then processed (scalinga and PCA) and then merged.
    '''
    data_selection, data_id, df_label = split_data(data) #Splits the data and the label

    data_scaled = scale_data(data_selection, scaler=preprocessing.StandardScaler())
    data_pca = pca_data(data_scaled)

    data_merged, df_label, df_processed = merge_data(data_selection, data_id, df_label, data_pca)
    #Merges the processed data and the label
    return data_merged, df_label, df_processed

def processing_data_rfecv(data):
    '''
    This function processes the data.
    First the data is splitted, then processed (scaling and rfecv) and then merged.
    '''
    data_selection, data_id, df_label = split_data(data) #Splits the data and the label

    data_scaled = scale_data(data_selection, scaler=preprocessing.StandardScaler())

    data_merged, df_label, df_processed = merge_data(data_selection, data_id, df_label, data_scaled)
    #Merges the processed data and the label
    data_rfecv = rfecv_data(df_processed,df_label) 
    return data_rfecv, df_label, df_processed