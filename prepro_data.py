'''
This module contains a function which splits the data into labels and features, 
performs preprocessing steps and merges it back.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import decomposition
from sklearn.feature_selection import VarianceThreshold
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
    df_processed = pd.DataFrame(data_processed, columns=data_selection.columns) #, columns=data_selection.columns
    df_processed.insert(0, 'ID', data_id.values)  #Inserts a column with the ID
    df_processed = df_processed.set_index(df_processed.columns[0]) #Sets the ID column as the index

    #Merge the label and processed data DataFrames
    data_merged = pd.merge(df_label, df_processed, on='ID')
    return df_label, df_processed

def merge_data_nonames(data_id, df_label, data_processed):
    '''
    This function merges the processed data and the labels.
    The column names are not added.
    '''
    #Create a dataframe with the processed data
    df_processed = pd.DataFrame(data_processed) #, columns=data_selection.columns
    df_processed.insert(0, 'ID', data_id.values)  #Inserts a column with the ID
    df_processed = df_processed.set_index(df_processed.columns[0]) #Sets the ID column as the index

    #Merge the label and processed data DataFrames
    data_merged = pd.merge(df_label, df_processed, on='ID')
    return df_label, df_processed


def scale_data(data_selection, scaler=preprocessing.StandardScaler()):
    '''
    This function scales the data.
    The default scaler is StandardScaler.

    It returns a dataframe with the merged data and the labels.
    '''
    scaler.fit(data_selection)
    data_scaled = scaler.transform(data_selection)
    return data_scaled, scaler


def variance_threshold(data, variance_threshold=0.1):
    '''
    This function removes features that have low variance.
    '''
    selector = VarianceThreshold(threshold=variance_threshold)  # Set the threshold
    data_threshold = selector.fit_transform(data)
    selected_columns = data.columns[selector.get_support()]

    return data_threshold, selector, selected_columns

def correlation_data(data, correlation_threshold=0.9):
    '''
    This function removes features that have a high correlation.
    '''
    df_data = pd.DataFrame(data)
    corr_matrix = df_data.corr().abs()

    to_drop = set()
    for column in corr_matrix.columns:
        for correlated_column in corr_matrix.columns:
            if column != correlated_column and corr_matrix.at[column, correlated_column] > correlation_threshold:
                to_drop.add(correlated_column)
                break
    data_correlation = df_data.drop(columns=to_drop)

    return data_correlation, to_drop


def pca_data(data):
    '''
    This function performs a PCA analysis on the data.
    The number of components is such that the 95% of the variance is retained.
    '''
    pca = decomposition.PCA(n_components=0.95)
    pca.fit(data)
    data_pca = pca.transform(data)

    return data_pca, pca

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
        min_features_to_select=15) #Dit nu een beetje gegokt
    labels = labels.values.ravel() #Makes a 1d array
    rfecv.fit(data, labels) #Fits the rfecv

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])

    selected_features = data.columns[rfecv.support_] #Gets the best number of features
    data_rfecv = pd.DataFrame(rfecv.transform(data), columns=selected_features, index=data.index)
    return data_rfecv, rfecv, selected_features


def processing_data_scaling(data):
    '''
    This function processes the data.
    First the data is splitted, then processed (scaling) and then merged.
    '''
    data_selection, data_id, df_label = split_data(data) #Splits the data and the label

    data_threshold, variance_data, selected_columns = variance_threshold(data_selection) #Variance threshold
    data_variance = pd.DataFrame(data_threshold, columns=selected_columns, index=data_selection.index) 
    #Creates a dataframe with original column names

    data_correlation, drop_correlation = correlation_data(data_variance) #Removes highly correlated data

    data_scaled, scaler = scale_data(data_correlation, scaler=preprocessing.MinMaxScaler()) #Scale the data

    df_label, df_processed = merge_data(data_correlation, data_id, df_label, data_scaled)
    #Merges the processed data and the label

    return df_label, df_processed, variance_data, selected_columns, drop_correlation, scaler

def processing_data_scaling_test(data_test, variance_data, selected_columns, drop_correlation, scaler):
    '''This function processes the data.
    The variance fit and scaler fit from the train data is used, and the selected columns of the correlation of the 
    train data is used.'''
    data_selection, data_id, df_label = split_data(data_test) #Splits the features and label

    data_test_variance = variance_data.transform(data_selection) #Fits the variance of the train data
    df_variance = pd.DataFrame(data_test_variance, columns=selected_columns, index=data_selection.index) #Creates a DF with correct column names

    data_test_correlation = df_variance.drop(columns=list(drop_correlation)) #Selects the correct columns of the train data

    data_test_scaled = scaler.transform(data_test_correlation) #Scales the data using the train data

    df_label, df_processed = merge_data(data_test_correlation, data_id, df_label, data_test_scaled) #Merges the data

    return df_label, df_processed


def processing_data_pca(data):
    '''
    This function processes the data.
    First the data is splitted, then processed (scalinga and PCA) and then merged.
    '''
    data_selection, data_id, df_label = split_data(data) #Splits the data and the label

    data_threshold, variance_data, selected_columns = variance_threshold(data_selection) #Variance threshold
    data_variance = pd.DataFrame(data_threshold, columns=selected_columns, index=data_selection.index) 
    #Creates a dataframe with original column names

    data_correlation, drop_correlation = correlation_data(data_variance) #Removes highly correlated data

    data_scaled, scaler = scale_data(data_correlation, scaler=preprocessing.StandardScaler()) #Scale the data

    data_pca, pca = pca_data(data_scaled) #Reduce dimensions

    df_label, df_processed = merge_data_nonames(data_id, df_label, data_pca)
    #Merges the processed data and the label
    return df_label, df_processed, variance_data, selected_columns, drop_correlation, scaler, pca

def processing_data_pca_test(data_test, variance_data, selected_columns, drop_correlation, scaler, pca):
    '''This function processes the data.
    The variance fit and scaler fit from the train data is used, and the selected columns of the correlation of the 
    train data is used. The PCA fit of the train data is also applied.'''
    data_selection, data_id, df_label = split_data(data_test) #Splits the features and label

    data_test_variance = variance_data.transform(data_selection) #Fits the variance of the train data
    df_variance = pd.DataFrame(data_test_variance, columns=selected_columns, index=data_selection.index) #Creates a DF with correct column names

    data_test_correlation = df_variance.drop(columns=list(drop_correlation)) #Selects the correct columns of the train data

    data_test_scaled = scaler.transform(data_test_correlation) #Scales the data using the train data

    data_test_pca = pca.transform(data_test_scaled) #Fits the PCA

    df_label, df_processed = merge_data_nonames(data_id, df_label, data_test_pca) #Merges the data

    return df_label, df_processed


def processing_data_rfecv(data):
    '''
    This function processes the data.
    First the data is splitted, then processed (scaling and rfecv) and then merged.
    '''
    data_selection, data_id, df_label = split_data(data) #Splits the data and the label

    data_threshold, variance_data, selected_columns = variance_threshold(data_selection) #Variance threshold
    data_variance = pd.DataFrame(data_threshold, columns=selected_columns, index=data_selection.index) 
    #Creates a dataframe with original column names

    data_correlation, drop_correlation = correlation_data(data_variance) #Removes highly correlated data

    data_scaled, scaler = scale_data(data_correlation, scaler=preprocessing.StandardScaler()) #Scale the data

    df_label, df_processed = merge_data(data_correlation, data_id, df_label, data_scaled)
    #Merges the processed data and the label
    data_rfecv, rfecv, selected_features = rfecv_data(df_processed,df_label)
    return data_rfecv, df_label, variance_data, selected_columns, drop_correlation, scaler, rfecv, selected_features

def processing_data_rfecv_test(data_test, variance_data, selected_columns, drop_correlation, scaler, rfecv, selected_features):
    '''This function processes the data.
    The variance fit and scaler fit from the train data is used, and the selected columns of the correlation of the 
    train data is used.
    The RFECV fit of the train data is applied'''
    data_selection, data_id, df_label = split_data(data_test) #Splits the features and label

    data_test_variance = variance_data.transform(data_selection) #Fits the variance of the train data
    df_variance = pd.DataFrame(data_test_variance, columns=selected_columns, index=data_selection.index) #Creates a DF with correct column names

    data_test_correlation = df_variance.drop(columns=list(drop_correlation)) #Selects the correct columns of the train data

    data_test_scaled = scaler.transform(data_test_correlation) #Scales the data using the train data

    df_label, df_processed = merge_data(data_test_correlation, data_id, df_label, data_test_scaled) #Merges the data

    data_test_rfecv = pd.DataFrame(rfecv.transform(df_processed), columns=selected_features, index=df_processed.index) #Fits the RFECV

    return data_test_rfecv, df_label