'''
This module contains a function which scales the data.
'''

from sklearn import preprocessing
import pandas as pd

def scale_data(data, scaler=preprocessing.StandardScaler()):
    '''
    This function scales the data.
    The default scaler is StandardScaler.

    It returns a dataframe with the merged data and the labels.
    '''
    #Create a dataframe with the label
    data_reset = data.reset_index() #Sets the index as the first column
    data_label = data_reset.iloc[:,0:2] #Gets a dataframe with the ID and the label
    df_label = data_label.set_index(data_label.columns[0]) #Sets the ID column as the index

    #Create a dataframe with the features
    data_id = data_reset.iloc[:,0] #Gets a dataframe with the ID
    data_selection = data_reset.iloc[:,2:] #Gets a dataframe with only the features

    #Scale the features
    scaler.fit(data_selection)
    data_scaled = scaler.transform(data_selection)

    #Create a dataframe with the scaled data
    df_scaled = pd.DataFrame(data_scaled, columns=data_selection.columns)
    df_scaled.insert(0, 'ID', data_id.values)  #Inserts a column with the ID
    df_scaled = df_scaled.set_index(df_scaled.columns[0]) #Sets the ID column as the index

    #Merge the label and scaled data DataFrames
    data_merged = pd.merge(df_label, df_scaled, on='ID')

    return data_merged, data_scaled, df_label