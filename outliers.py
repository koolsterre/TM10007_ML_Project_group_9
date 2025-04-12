import numpy as np
from scipy.stats import zscore
import pandas as pd

def outlier_detection(data):
    drempel = 3
    total_outliers = 0
    for column in data.select_dtypes(include=[np.number]).columns:

        z_scores = zscore(data[column])
        mean_value = data[column].mean()
        std_value = data[column].std()

        outliers = abs(z_scores) > drempel

    # Vervang de outliers met de grenswaarden
        data.loc[outliers,column] = np.clip(data.loc[outliers,column], mean_value - drempel * std_value, mean_value + drempel * std_value)
        total_outliers += outliers.sum()

    return data, total_outliers


def outlier_detection_test(data_train, data_test):
    drempel = 3
    total_outliers_test = 0
    for column in data_train.select_dtypes(include=[np.number]).columns:

        mean_value = data_train[column].mean()
        std_value = data_train[column].std()
    
        z_scores = list()
        for i in data_test.index:
            data_value = data_test.loc[i,column]
            if std_value == 0:
                z_score = 0
            else:
                z_score = abs((data_value - mean_value)/std_value)
            z_scores.append(z_score)

        z_scores_series = pd.Series(z_scores,index=data_test.index) #Create a Series with the z-scores, using the mean and STD of train_data
        outliers = z_scores_series > drempel #Detecting outliers

    # Vervang de outliers met de grenswaarden
        data_test.loc[outliers,column] = np.clip(data_test.loc[outliers,column], mean_value - drempel * std_value, mean_value + drempel * std_value)
        total_outliers_test += outliers.sum()

    return data_test, total_outliers_test
