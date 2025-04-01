import numpy as np
from scipy.stats import zscore

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