# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 20:02:56 2026

@author: Admin
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

#read the dataset
df=pd.read_csv('crime_data.csv')
df.head()
print("Shape of Dataset is:",df.shape)
#Shape of Dataset is: (50,5)

#Check Missing Values
print(df.isnull().sum())
#No NUll values presen in dataset

#Check Datatype of each column
print(df.dtypes)
'''
Unnamed: 0        str
Murder        float64
Assault         int64
UrbanPop        int64
Rape          float64
dtype: object
'''
#Seperate  the numerical and categorical columns
cat_cols = df.select_dtypes(include='object').columns
cat_cols
#No categorical columns presen in this dataset
num_cols = df.select_dtypes(include=['int64','float64']).columns
num_cols

#Removing duplicates
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)  #(302,14)
#No duplicate records in Dataset

#Outlier Treatment
for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.xlabel(col)
    plt.show()
    
#outlier present in rape column
 
from feature_engine.outliers import Winsorizer
# For 'Rape'
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['Rape']
         )
    
df_t=winsor.fit_transform(df[['Rape']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['Rape'])

###############################################
#Normalization
# Make a copy for demonstration
df_scaled = df.copy()

# Normalization (min = 0, max = 1)
'''It usually transforms values into a range of 0 to 1.'''

scaler_minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[num_cols] = scaler_minmax.fit_transform(df_minmax[num_cols])
##################################################################
from scipy.stats import skew

# Import PowerTransformer (not used directly here, but commonly used for Box-Cox)
from sklearn.preprocessing import PowerTransformer

# Calculate skewness of the numeric column
# dropna() removes missing values because skew() cannot work with NaNs

for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col],kde=True)
    #sns.histplot(df1[col], kde=True)\
    #.set_title("Final Transformed")
    plt.show()
    #fbs,sex, exang,
#df1=pd.DataFrame()
for col in num_cols:
    skew_val=df[col].skew()
    if skew_val > 0:
        
        # Inform user that right skew is detected
        print("Positive skew detected → Applying Log transform")
        
        # Apply log(1 + x) transformation
        # log1p is used to safely handle zero values
        # This compresses large values on the right side
        df[col] = np.log1p(df[col])

    # If skewness is less than 0, data is negatively (left) skewed
    elif skew_val < 0:
        
        # Inform user that left skew is detected
        print("Negative skew detected → Applying Reflection + Log")
        
        # Reflect the data to convert left skew into right skew
        # max() + 1 ensures all reflected values are positive
        reflected = df[col].max() + 1 - df[col]
        
        # Apply log transformation after reflection
        df[col] = np.log1p(reflected)

    # If skewness is approximately zero, data is already symmetric
    else:
        
        # Inform user that no transformation is needed
        print("No significant skew → No transformation applied")
        
        # Keep original values unchanged
        df[col] = df[col]
# Plot the final transformed distribution
# KDE curve helps visualize smooth distribution
for col in num_cols:
    plt.figure(figsize=(8,4))
    #sns.histplot(df[col],kde=True)
    sns.histplot(df[col], kde=True)\
    .set_title("Final Transformed")
    plt.show()
########################################################################   
# Zero Variance Feature Removal
# Create constant column to demonstrate
#What is a zero-variance feature?

from sklearn.feature_selection import VarianceThreshold

# Create the VarianceThreshold object
selector = VarianceThreshold(threshold=0.0)  # Remove features with 0 variance

# Apply only to numeric features

df_var = selector.fit_transform(df[num_cols])

#selector.fit_transform(...) removes any column with variance = 0
print("Shape  of orginal dataframe:", df_var.shape)
#(50,4)
#No columns with Zeo Variance
########################################################################
#there is no categorical feature in dataset so no need of one hot encoding
#################################################################

        
for col in num_cols:
   print(df.value_counts().sum())
# No Target column present in the dataset so no need to do balance

###########################################################################




