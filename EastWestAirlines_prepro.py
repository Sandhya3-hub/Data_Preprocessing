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
#######################################################################
# 1. BUSINESS UNDERSTANDING
####################################################################
'''The EastWestAirlines dataset is commonly used for customer segmentation in the airline industry. It contains information about airline customers, their mileage balances, flight activity, and frequent-flyer program participation.
The airline wants to understand customer travel behavior and identify different groups of customers based on their loyalty, flight activity, and accumulated miles.
'''
#Data Uderstanding
'''
ID#	       Unique customer identifier	   Identifies each customer; 
Balance	  Current mileage balance	    Indicates accumulated miles 
                                        and possible customer value
Qual_miles	 Qualifying miles earned   Helps assess eligibility for                      frequent-flyer status
cc1_miles	Miles earned through 
credit card 1	Represents credit-card-related activity
cc2_miles	Miles earned through credit card 2	Represents activity through another card category
cc3_miles	Miles earned through credit card 3	Represents activity through a third card category
Bonus_miles	Miles earned from non-flight bonus activities	Indicates participation in bonus programs
Bonus_trans	Number of bonus transactions	Measures engagement with bonus programs
Flight_miles_12mo	Flight miles accumulated in the last 12 months	Measures recent flying activity
Flight_trans_12	Number of flight transactions in the last 12 months	Indicates recent travel frequency
Days_since_enroll	Days since enrollment in the loyalty program	Indicates customer tenure
'''
######################################################################
#     2. EXPLORATOER DATA ANALYSIS
######################################################################
#read the dataset
df=pd.read_excel('EastWestAirlines.xlsx')
df.head()
print("Shape of Dataset is:",df.shape)
#Shape of Dataset is: (3999,12)

#Check Missing Values
print(df.isnull().sum())
#No NUll values presen in dataset

#Check Datatype of each column
print(df.dtypes)
'''
Balance        int64    Qual_miles      int64    cc1_miles         int64
cc2_miles      int64   cc3_miles        int64    Bonus_miles       int64
Bonus_trans    int64   Flight_miles_12mo    int64 
Flight_trans_12   int64 Days_since_enroll  int64  Award?          Int64
dtype: object

'''
df["Award?"].value_counts()
sns.countplot(data=df, x="Award?")
plt.show()
#Class 0 has 2500 and class 1 has 1400 count so lcass in imbalanced


########################Business  moment decision#####################

# 1st Moment: Mean(central Tendency)

mean_val=df.mean(numeric_only=True)
print("\n Mean(First Movement):\n",mean_val)
#inference: mean gives the average measurement

#2. Second Moment: Varience/Standard Daviation(Dispersion)
var_values=df.var(numeric_only=True)
std_values=df.std(numeric_only=True)
print("\n Variance(Second Movement):\n",var_values)
print("\n Standard Daviation(Second Movement):\n",std_values)
num_cols = df.select_dtypes(include=['int64','float64']).columns
#3rd and 4th Moment
#skewness and kurtosis
print("Skewness:")
print(df[num_cols].skew())
'''Mileage and transaction features may be highly right-skewed because some customers travel much more frequently than others.'''
print("Kurtosis:")
print(df[num_cols].kurtosis())
'''
Positive excess kurtosis indicates heavier tails or a more pronounced peak than a normal distribution.

Negative excess kurtosis indicates lighter tails or a flatter distribution.
'''

#Univariate analysis
df[num_cols].hist(figsize=(16, 12), bins=30)
plt.tight_layout()
plt.show()
#Histograms show the distribution, concentration, skewness, and possible extreme values in customer mileage and travel activity.

#Box plots to identify outliers
for col in num_cols:
    if col != "ID#":
        plt.figure(figsize=(7, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Box Plot of {col}")
        plt.show()
'''Inference: Box plots help identify potential outliers in features such as Balance, Bonus_miles, and Flight_miles_12mo. These values should be investigated before deciding whether to cap or remove them.'''

#Correlation analysis
plt.figure(figsize=(14, 10))

sns.heatmap(
    df[num_cols]
      .drop(columns=["ID#"], errors="ignore")
      .corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

'''
The heatmap helps to identify relationships between mileage balances, bonus miles, flight transactions, and enrollment duration.There is a strong relationship between cc1_miles and bonus_miles,bonus_trans. Strong correlations may indicate overlapping information among features.'''

#Relationship between important features
sns.scatterplot(
    data=df,
    x="Bonus_miles",
    y="Balance"
)
plt.title("Bonus Miles vs Balance")
plt.show()

sns.scatterplot(
    data=df,
    x="Flight_miles_12mo",
    y="Flight_trans_12"
)
plt.title("Flight Miles vs Flight Transactions")
plt.show()

'''These plots help to determine whether customers who accumulate more bonus miles tend to have medium  balances, and flight mileage is positively associated with the number of flight transactions.As flight milege increase transaction also increases.'''


#########################################################################       3. DATA PREPROCESSING
#################################################################
#Seperate  the numerical and categorical columns
cat_cols = df.select_dtypes(include='object').columns
cat_cols
#No categorical columns presen in this dataset
num_cols=num_cols.drop('ID#')
num_cols
#Removing duplicates
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)
#No  duplicates in Dataset

#Outlier Treatment

from feature_engine.outliers import Winsorizer    
Outlier=['Balance','Qual_miles','cc2_miles','cc3_miles','Bonus_miles','Bonus_trans','Flight_miles_12mo','Flight_trans_12']
exclude_cols=['Qual_miles','cc2_miles','cc3_miles']

cols_to_cap = [col for col in num_cols if col not in exclude_cols]
for col in cols_to_cap: 
    winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=[col]
         )
    
    df_t=winsor.fit_transform(df[[col]])
#sns.boxplot(df['Sales'])
    sns.boxplot(df_t[col])
'''
#Qual_miles
winsor= Winsorizer(
           capping_method= 'quantiles',
           tail='both',
           variables=['Qual_miles']
           )
      
df_t=winsor.fit_transform(df[['Qual_miles']])
  #sns.boxplot(df['Sales'])
sns.boxplot(df_t['Qual_miles'])

#sns.boxplot(df_t)

# cc2_miles
winsor= Winsorizer(
         capping_method= 'quantiles',
         tail='both',
         variables=['cc2_miles']
         )
    
df_t=winsor.fit_transform(df[['cc2_miles']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['cc2_miles'])

#cc3_miles
winsor= Winsorizer(
         capping_method= 'quantiles',
         tail='both',
         fold=1.5,
         variables=['cc3_miles']
         )
    
df_t=winsor.fit_transform(df[['cc3_miles']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['cc3_miles'])

#Error: Input columns ['cc2_miles'] have low variation for method 'quantiles'.
'''
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
#(3999,11)
#No columns with Zeo Variance
########################################################################

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
#Qual_miles,cc2_miles,cc3_miles  column have values >0.85
#Need to balance imbalance dataset So need to use SMOTE 
###########################################################################
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
#SMOTE
# Split features and target
#df = df.replace([np.inf, -np.inf], np.nan) #replace infinity values with NaN
df['Award?'] = df['Award?'].astype(int)
df['Award?'].value_counts()
#for checking wheather target class only one class ?
print(df['Award?'].nunique())
print("Original target classes:")
print(df['Award?'].value_counts(dropna=False))
#  Target has only one class. In order to make dataset balance Target variable should have atleast two classes. So Cant make data Balanced.


##############################################


