'''
This module contains functions to identify and replace outliers in a dataset using the Interquartile Range (IQR) method.
The `replace_outliers` function identifies outliers in the training data and replaces them with the lower and upper bounds.
The `replace_outliers_test` function applies the same bounds to the test data.
'''

def replace_outliers(data_train):
    '''
    This function replaces outliers in the training data with the lower and upper bounds.
    It uses the IQR method to calculate the bounds and replaces values outside of these bounds.
    The function returns the modified training data, the total number of capped values, and the lower and upper bounds.
    '''
    #Select only numeric columns
    numeric_cols = data_train.select_dtypes(include=['number'])
    total_capped = 0
    #Change outlier in the dataframe
    for col in numeric_cols.columns:
        data = data_train[col].dropna()
 
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
 
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
 
        #Boolean mask for outliers
        lower_outliers = data_train[col] < lower_bound
        upper_outliers = data_train[col] > upper_bound
        capped_count = lower_outliers.sum() + upper_outliers.sum()
 
        #Update total
        total_capped += capped_count
 
        #Limit extreme values to lower and upper bound
        data_train[col] = data_train[col].apply(lambda x: 
            lower_bound if x < lower_bound else
            upper_bound if x > upper_bound else
            x
        )
    return data_train, total_capped, lower_bound, upper_bound
 
def replace_outliers_test(data_train, data_test, lower_bound, upper_bound):
    '''
    This function replaces outliers in the test data with the lower and upper bounds.
    It uses the IQR method to calculate the bounds based on the training data and replaces values outside of these bounds.
    The function returns the modified test data and the total number of capped values.
    '''
    #Select only numeric columns
    numeric_cols = data_test.select_dtypes(include=['number'])
    total_capped_test = 0
    #Change outlier in the dataframe
    for col in numeric_cols.columns:
        data = data_train[col].dropna() #Uses the training data to calculate the bounds
 
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
 
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
 
         #Boolean mask for outliers
        lower_outliers = data_test[col] < lower_bound #Applies the bound to the test data
        upper_outliers = data_test[col] > upper_bound
        capped_count = lower_outliers.sum() + upper_outliers.sum()
 
        #Update total
        total_capped_test += capped_count
 
        #Limit extreme values to lower and upper bound
        data_test[col] = data_test[col].apply(lambda x: 
            lower_bound if x < lower_bound else
            upper_bound if x > upper_bound else
            x
        )
    return data_test, total_capped_test