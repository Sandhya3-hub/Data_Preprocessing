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
df=pd.read_csv('mtcars.csv')
df.head()
print("Shape of Dataset is:",df.shape)
#Shape of Dataset is: (32,11)

#Check Missing Values
print(df.isnull().sum())
#No NUll values presen in dataset

#Check Datatype of each column
print(df.dtypes)
'''
mpg     float64    cyl       int64     disp    float64
hp        int64    drat    float64     wt      float64
qsec    float64    vs        int64     am        int64
gear      int64     carb      int64
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
print("After removing duplicates:", df.shape)
#No  duplicates in Dataset

#Outlier Treatment
for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.xlabel(col)
    plt.show()
    
 #Outliers: mpg,hp,wt,qsec,carb
from feature_engine.outliers import Winsorizer    
Outlier=['mpg','hp','wt','qsec','carb']

for col in Outlier: 
    winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=[col]
         )
    
    df_t=winsor.fit_transform(df[[col]])
#sns.boxplot(df['Sales'])
    sns.boxplot(df_t[col])


###############################################
#Normalization and Standardization
# Make a copy for demonstration
df_scaled = df.copy()

# Standardization (mean = 0, std = 1)
'''
It transforms values so that the feature has a mean of approximately 0 and 
a standard deviation of 1.
'''
scaler_std = StandardScaler()
df_scaled[num_cols] = scaler_std.fit_transform(df_scaled[num_cols])

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
#(3999,11)
#No columns with Zeo Variance
########################################################################
#No categorical columns present so,
#no need to do one hot encodeing as 
#################################################################
#Checking  if is Data Imbalance or not?
# 1. Check for highly skewed features
#new_num_cols = [col for col in num_cols if col != 'pdays']
skewness = df[num_cols].skew()
high_skew = skewness[abs(skewness) > 3] # Common threshold for extreme skew
print("Highly skewed features:\n", high_skew)
# Highly skewed features:
#Qual_miles     3.914590
#cc2_miles     10.555336
#cc3_miles     16.227604
#dtype: float64

# 2. Check for zero-inflation or single-value dominance
for col in df.columns:
    top_value_pct = df[col].value_counts(normalize=True).iloc[0]
    print(top_value_pct)
    if top_value_pct > 0.85: # If one value dominates > 85% of the data
        print(f"Feature '{col}' is heavily imbalanced. Top value occupies {top_value_pct*100:.2f}% of data.")
# No need to do balancing as no column has op_value>.85
###########################################################################
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
#SMOTE
# Split features and target
#df = df.replace([np.inf, -np.inf], np.nan) #replace infinity values with NaN

#for checking wheather target class only one class ?
print(df['Award?'].nunique())
print("Original target classes:")
print(df['Award?'].value_counts(dropna=False))
#  Target has only one class. In order to make dataset balance Target variable should have atleast two classes. So Cant make data Balanced.


##############################################


