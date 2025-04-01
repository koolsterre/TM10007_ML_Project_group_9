import numpy as np
from scipy.stats import zscore

def outlier_detection(data):
    drempel = 3
    total_outliers = 0
    for column in data.select_dtypes(include=[np.number]).columns:
    # Bereken de Z-scores voor de huidige kolom
        z_scores = zscore(data[column])
        mean_value = data[column].mean()
        std_value = data[column].std()

    # Pas de Z-score gebaseerde cap toe direct op de huidige kolom
        outliers = abs(z_scores) > drempel

    # Cap de outliers door ze te vervangen met de grenswaarde
        data.loc[outliers,column] = np.clip(data.loc[outliers,column], mean_value - drempel * std_value, mean_value + drempel * std_value)

    # Tel het aantal gecapte outliers voor de huidige kolom
        total_outliers += outliers.sum()

    return data, total_outliers