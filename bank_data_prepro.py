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

######################################################################
#               1. DATA UNDERSTANDING
####################################################################
#Business Understanding
''''The bank dataset is used to understand customer characteristics, financial behavior, and the factors that influence customers to subscribe to a bank's product or service. 
In a typical bank_data dataset, the main objective is to identify customers who are more likely to subscribe to a term deposit after a marketing campaign.
'''
#Data Understanding
'''
age            int64       default      int64      balance         int64   housing        int64
loan           int64      duration       int64     
campaign       int64     pdays           int64
previous       int64     poutfailure     int64     
poutother      int64     poutsuccess     int64
poutunknown    int64     con_cellular    int64     
con_telephone  int64    con_unknown       int64
divorced       int64    married           int64   
single         int64  joadmin.          int64
joblue.collar  int64    joentrepreneur     int64   
johousemaid     int64   jomanagement       int64
joretired      int64    joself.employed   int64 
 joservices    int64   jostudent          int64
jotechnician   int64   jounemployed       int64  
jounknown        int64   y                  int64
'''

######################################################################
#               2. EXPLORATORY DATA ANALYSIS
#######################################################################
#read the dataset
df=pd.read_csv('bank_data.csv')
df.head()
print("Shape of Dataset is:",df.shape)
#Shape of Dataset is: (45211,32)

#Check Missing Values
print(df.isnull().sum())
#No NUll values presen in dataset

#Check Datatype of each column
print(df.dtypes)
'''
age            int64       default      int64      balance         int64   housing        int64
loan           int64      duration       int64      campaign        int64   pdays          int64
previous       int64     poutfailure      int64     poutother        int64   poutsuccess    int64
poutunknown    int64    con_cellular    int64      con_telephone     int64   con_unknown    int64
divorced       int64     married          int64    single           int64    joadmin.       int64
joblue.collar  int64    joentrepreneur     int64   johousemaid     int64   jomanagement       int64
joretired      int64    joself.employed   int64  joservices       int64   jostudent          int64
jotechnician   int64   jounemployed       int64  jounknown        int64   y                  int64
dtype: object
'''

df.describe().iloc[:,2:5]
'''
               age       default  
count  45211.000000  45211.000000  
mean      40.936210      0.018027  
std       10.618762      0.133049  
min       18.000000      0.000000  
25%       33.000000      0.000000  
50%       39.000000      0.000000  
75%       48.000000      0.000000  
max       95.000000      1.000000  

            balance       housing          loan
count   45211.000000  45211.000000  45211.000000
mean     1362.272058      0.555838      0.160226
std      3044.765829      0.496878      0.366820
min     -8019.000000      0.000000      0.000000
25%        72.000000      0.000000      0.000000
50%       448.000000      1.000000      0.000000
75%      1428.000000      1.000000      0.000000
max    102127.000000      1.000000      1.000000
#Age: The average customer age range is 41. and 50% customers are in range of 33 to 39. 
Mean is slightly greater than median so slightly right skewed.
#default column: mean is recore=ded as 0.018=1.80%. it is a binary categorical column.
                1 for yes and 0 for No.column whih is binary where median =0 so no skewness in this column
#balance: Mean = 1,362.27 and median = 448: 
    The mean is much higher than the median, suggesting a right-skewed distribution.
    some customers have minimum negative balance and some have exceptionally high balance.
    
    
#Housing: Approximately 55.58% of clients have a housing loan, while 44.42% do not. 
         The distribution is relatively balanced between the two categories, 
         with a slightly higher proportion of clients having housing loans.
         
#Loan: Mean = 0.160: Approximately 16.02% of clients have a personal loan.
      Median = 0: More than half of the clients do not have a personal loan.
     The feature is dominated by the 0 category, indicating a class imbalance between clients 
     with and without personal loans.
 '''

df.describe().iloc[:,5:8]
print(df.isnull().sum())
#NO  missing values in dataset
#Removing duplicates
df.drop_duplicates(inplace=True)
print("After removing duplicates:",df.shape)  #After removing duplicates: (1000, 10)
#No duplicate values in dataset

#--------------Business Moment Decision-----------------
# 1st Moment: Mean(central Tendency)

mean_val=df.mean(numeric_only=True)
print("\n Mean(First Movement):\n",mean_val)
#inference: mean gives the average measurement
'''
The average age of clients is approximately 41 years, and the average account balance 
is around 1,362.30. The average call duration is approximately 258 seconds, 
while the average number of campaign contacts is about 2.76. 
Around 55.59% of clients have a housing loan, and 16.02% have a personal loan. 
Only 1.80% of clients have a credit default.
Regarding previous campaign outcomes, 81.75% of clients have an unknown previous outcome, 
while 10.84% experienced failure and 3.34% experienced success. 
Cellular communication is the most common contact method, 
accounting for approximately 64.77% of clients.
In terms of marital status, 60.19% of clients are married, 28.29% are single, 
and 11.52% are divorced. 
The most common job categories are blue-collar workers (21.53%), 
management (20.92%), and technicians (16.80%).
'''
#2. Second Moment: Varience/Standard Daviation(Dispersion)
var_values=df.var(numeric_only=True)
std_values=df.std(numeric_only=True)
print("\n Variance(Second Movement):\n",var_values)
print("\n Standard Daviation(Second Movement):\n",std_values)

#Inference: varience shows  how widely the feature values are spread. Age has wide spread than other features  
#higher Standard deviation shows higher variance and lower value shows values are 
#more constrated around mean
'''
std:
1. age(10.62):moderate variation in client age
balance(3044):substantial differences in account balances among clients. 
The housing feature has a standard deviation of 0.497, reflecting its relatively balanced distribution, 
whereas loan has a standard deviation of 0.367, consistent with fewer clients having personal loans.
The target feature y has a standard deviation of 0.321, reflecting the imbalance between 
clients who subscribed to a term deposit and those who did not. 
Most categorical features have standard deviations below 0.5 because they are one-hot encoded 
or binary variables.'''


#Third Movement: Skewness(Symmentry)
skew_val=df.skew(numeric_only=True)
print("\nSkewness(Third Moment)\n:",skew_val)
#inference:It helps us understand whether data is concentrated on one side and whether a longer
# tail extends toward the left or right.
'''
Skewness > 0: Positively skewed (right-skewed); the distribution has a longer right tail.
Skewness < 0: Negatively skewed (left-skewed); the distribution has a longer left tail.
Skewness ≈ 0: Approximately symmetric.

#age(0.68): indicating a moderately right-skewed distribution. 
#balance has a high positive skewness of 8.36, suggesting that a small number of clients
have exceptionally high account balances. 
#duration (3.14), campaign (4.90), and pdays (2.62): positively skewed: 
        indicating the presence of some unusually large values.
#The previous (41.85): extremely high skewness, 
suggesting that most clients have few or no previous contacts, while a small number have many.
#The default feature (7.25): strong positive skewness, 
        consistent with very few clients having a credit default.
#Among the categorical features, housing (-0.22) and married (-0.42):slight negative skewness,
#while single (0.96) and con_unknown (0.94): positive skewness. 
#The encoded categories with high positive skewness,e.g. jounknown (12.41) and jostudent (6.72), 
generally indicate that these categories occur relatively infrequently.

'''

#4th moment: Kurtosis(peakness)
kurt_val=df.kurtosis(numeric_only=True)
print("\n Kurtosis(Forth Moment):\n",kurt_val)
#Kurtosis measures the heaviness of a distribution's tails and helps 
#us to understand how likely extreme values are compared 
#with a normal distribution.
'''
Kurtosis = 0: Similar tail heaviness to a normal distribution.
Kurtosis > 0: Leptokurtic — heavier tails and potentially more extreme values.
Kurtosis < 0: Platykurtic — lighter tails and generally flatter distributions.

Feature	   Kurtosis	        Inference
age	       0.32	      Close to a normal distribution in terms of kurtosis, with slightly heavier tails.
default	  50.50	      Extremely high kurtosis, likely due to the binary nature of the feature and a strong imbalance between its values.
balance	  140.75	 Extremely heavy tails, indicating potential extreme account balances and outliers.
housing	 -1.95	     Very low kurtosis, consistent with a binary feature.
loan	  1.43	     Positive kurtosis, suggesting a tendency toward extreme values; binary coding also affects this measure.
duration 18.15	     Heavy tails, suggesting some unusually long call durations.
campaign 39.25	    Very heavy tails, indicating some clients received substantially more campaign contacts than others.
pdays	 6.93	    Heavy tails, suggesting extreme values in the number of days since the previous contact.
previous 4506.77	Extremely high kurtosis, likely caused by many small values and a few unusually large values.

'''
#Univarient Analysis

sns.histplot(df.age,kde=True)  
'''
The histogram shows a roughly bell-shaped curve(unimodel)
somewhat close to a normal distribution,    
but slightly rigth-skewed(longer tail on the right)
This means that there some older users than the  main constration of users.
'''
 
sns.histplot(df.duration,kde=True)  
'''
The histogram shows a roughly bell-shaped curve(unimodel)
but slightly right-skewed(longer tail on the right)
The distribution has a longer tail toward lower income values. 
Means most of user have larger income '''

sns.histplot(df.balance,kde=True)  
'''
The data is heavily right-skewed (positively skewed).
the frequency count peaks dramatically at around 14,000 for the histogram bin 
and over 6,000 for the density line.
The highest peak occurs early on, roughly between 100 and 200 units, 
where the count reaches over 1,750 occurrences.
The graph has a long tail stretching to the right, all the way up to 5,000 units.
After the initial peak, the frequency drops drastically. 
By the time duration reaches 1,000 units, the count is extremely close to zero.
'''

#Scatter plot:Bivariate analysis

#Analyze the relationship between two features to understand how they are associated.
sns.scatterplot(data=df,x='balance',y='campaign',hue='y')
plt.title('Scatter Plot: Balance Vs Campaign')
plt.show() 
'''
• X-axis (balance): Represents the account balance of the individuals, ranging from negative values up to roughly 100,000.
• Y-axis (campaign): Represents the number of times a customer was contacted during this specific marketing campaign, ranging from 1 to over 60 times.

• High Density at Low Balances: The vast majority of the data points are heavily clustered near 
the lower end of the balance scale (between 0 and 20,000) and are contacted fewer than 20 times.
• Diminishing Returns on Contact Frequency: Orange dots (y = 1) are primarily concentrated at the bottom 
of the graph, mostly below 10–15 campaign contacts.
 When a customer is contacted an excessive number of times (e.g., more than 20 or 30 times, 
extending up to 60), almost all outcomes are blue (y = 0). This suggests that over-contacting a 
customer does not lead to a successful outcome.
• Outliers: There are a few notable outliers, such as a customer with a balance near 57,000 
who was contacted over 30 times (resulting in a 0), and a few high-balance individuals 
(above 80,000) who mostly received very few contacts(resulting in a 1).

'''
sns.scatterplot(data=df,x='age',y='duration',hue='y')
plt.title('Scatter Plot: Age Vs Duration')
plt.show() 
'''
Displays the relationship between a client's age and the duration of an interaction
majority of the data points are heavily concentrated between the ages of 25 and 60.
sharp drop in the density of interactions right after age 60, indicating far fewer clients in the older demographic.
clustering heavily below 1,500 units (likely seconds) across all age groups.
Long-duration interactions (above 2,000 units) primarily occur within the core 30–60 age range, 
with extreme outliers peaking near 5,000 units around age 45 and 60.
For durations under 2,000 units, successful (1 / orange) and unsuccessful (0 / blue) outcomes 
are completely intertwined, meaning age and short durations alone do not easily 
separate the two classes.
'''

#Box Plot 

# Select numerical features
features = [
    'age',
    'duration',
    'balance',
    'loan',
    'y'
]

# Create separate box plots
for feature in features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.xlabel(feature)
    plt.show()
'''
->Age feature has outlier and the middle 50% of the values, 
between the first quartile (Q1) and third quartile (Q3) 
that is in range of age 32 to 48
-> Duration feature:
    Middle 50%: The box extends approximately from ₹100(Q2) 
    to 300 (Q3).This indicates that the vast majority of events 
    have a short duration.
    •There is a significant number of ouliers 
    beyond the right whisker (which ends around 600).
    While most durations finish under 700 units, 
    extreme cases extend all the way up to nearly 5,000 units.
-> balance:The distribution is heavily skewed to the right.
        The vast majority of the data points are tightly clustered 
        below or just above 0, while a long tail of data points 
        extends far toward the positive end, reaching up to over 100,000.
        is extremely narrow and located very close to 0. 
        This means the typical variation in balance for the vast majority
        of the population is very small, and most individuals have a 
        relatively low balance.There is a very large number of 
        individual points plotted beyond the right whisker(1000 to extend upto 100000)
            
->Loan:
    The vast majority of the data points are located exactly at 0.0,
    represented by the thick vertical line on the far left.
    he data point at 1.0 is plotted as an individual circle, 
    indicating it is treated as an outlier relative to the rest of 
    the distribution.
    The dataset is heavily skewed toward 0.0. This implies that most 
    individuals or observations in this dataset do not have a loan.
'''

sns.boxplot(data=df,x='loan', y='age')

plt.xlabel('Loan (0=No, 1 = Yes)')
plt.ylabel('Age')
plt.show()
'''
Both plots are identicalw.r.to IQR and central tendency.
The most significant difference is at the 
upper end of the age spectrum.
Age does not appear to significantly influence
whether someone has a loan for the vast 
majority of the population, as the core 
distributions are nearly indistinguishable.
'''

#Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df[features].corr()
            ,annot=True,cmap='coolwarm')        
plt.title(" features correlation heatmap")
plt.show()

'''
-> The feature duration has the highest positive correlation with y at 0.39.
 This suggests that as the duration increases, the target variable y is 
 more likely to be positive or higher, though a value of 0.39 represents 
 a moderate positive relationship rather than a strong one.
-> Both age (0.025) and balance (0.053) have near-zero positive correlations 
with y. The feature loan has a very weak negative correlation with y at -0.068.
These values indicate that changes in these features have almost no linear
 impact on y.
• Balance and Loan (-0.084): This is the strongest negative correlation in 
the matrix.It implies that individuals with higher balances are marginally 
less likely to have a loan (or have smaller loan amounts)
• Age and Loan (-0.016): An almost non-existent negative relationship between age and having a loan.
• Duration and Loan (-0.012): Virtually zero correlation between interaction duration and loan status.
• Age and Duration (-0.0046): The closest value to absolute zero in the matrix, indicating that age has no bearing on contact duration.

'''

#CDF and PDF

#num_cols=df.select_dtypes(include=np.number).columns.drop('id')
#num_cols = df.select_dtypes(include=np.number).columns.drop('id')
num_cols = df[features]
for col in num_cols:
     plt.figure(figsize=(12,5))

        #pdf
     plt.subplot(1,2,1)
     sns.kdeplot(df[col],fill=True,color='blue')
     plt.title(f"PDF of {col}")
     plt.xlabel(col)
     plt.ylabel('Density')
     #CDF
     plt.subplot(1,2,2)
     sorted_vals=np.sort(df[col])
     y_vals=np.arange(len(sorted_vals))/float(len(sorted_vals))
     plt.plot(sorted_vals,y_vals,marker='.',linestyle='none',color='green')
     plt.title(f"CDF of {col}")
     plt.xlabel(col)
     plt.ylabel('Cummulative Probability')
     plt.tight_layout()
     plt.show()
     
'''

    
    
'''    

###################################################################
#               3. DATA PREPROCESSING
######################################################################

#Seperate  the numerical and categorical columns
binary_cat_cols = [
    col for col in df.columns 
    if df[col].dtype == 'int64' and set(df[col].dropna().unique()).issubset({0, 1})
]
binary_cat_cols
#No categorical columns presen in this dataset
num_cols = ['age','previous','pdays','campaign','balance','duration']
num_cols

#Removing duplicates
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)
#One record found as duplicate in Dataset

#Outlier Treatment
for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.xlabel(col)
    plt.show()
    
#age,previous,pdays,campaign,balance,duration has heavy outliers
 
from feature_engine.outliers import Winsorizer
# For 'age'
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['age']
         )
    
df_t=winsor.fit_transform(df[['age']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['age'])

#previous
winsor= Winsorizer(
           capping_method= 'quantiles',
           tail='right',
           variables=['previous']
           )
      
df_t=winsor.fit_transform(df[['previous']])
  #sns.boxplot(df['Sales'])
sns.boxplot(df_t['previous'])

#sns.boxplot(df_t)

# pdays
winsor= Winsorizer(
         capping_method= 'quantiles',
         tail='right',
         variables=['pdays']
         )
    
df_t=winsor.fit_transform(df[['pdays']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['pdays'])

#campaign,balance,duration
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['campaign']
         )
    
df_t=winsor.fit_transform(df[['campaign']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['campaign'])

#balance
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['balance']
         )
    
df_t=winsor.fit_transform(df[['balance']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['balance'])

#duration
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['duration']
         )
    
df_t=winsor.fit_transform(df[['duration']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['duration'])
###############################################
# Standardization:
    #Standardization is generally a good choice for scaling numerical features, especially if we plan to use machine learning algorithms such as Logistic Regression, SVM, KNN, or PCA.
# Make a copy for demonstration
df_scaled = df.copy()

# Standardization (mean = 0, std = 1)
'''
It transforms values so that the feature has a mean of approximately 0 and 
a standard deviation of 1.
'''
scaler_std = StandardScaler()
df_scaled[num_cols] = scaler_std.fit_transform(df_scaled[num_cols])

##################################################################
from scipy.stats import skew

# Import PowerTransformer (not used directly here, but commonly used for Box-Cox)
from sklearn.preprocessing import PowerTransformer

# Calculate skewness of the numeric column
# dropna() removes missing values because skew() cannot work with NaNs

for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col],kde=True)
    #sns.histplot(df[col], kde=True)\
    #.set_title("Final Transformed")
    plt.show()
#df=pd.DataFrame()
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
df = df.replace([np.inf, -np.inf], np.nan)
# Apply only to numeric features

df_var=selector.fit_transform(df[num_cols])

#selector.fit_transform(...) removes any column with variance = 0
print("Shape  of orginal dataframe:", df_var.shape)
#(45210,6)
#No columns with Zeo Variance
########################################################################

#Checking  if Data  is balance or not?
# 1. Check for highly skewed features

df['y'].value_counts()

# Dataset is imbalanced
#y
#0    39921
#1     5289

#Need to balance the  imbalance dataset So need to use SMOTE 
###########################################################################
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
#SMOTE
# Split features and target
#df = df.replace([np.inf, -np.inf], np.nan) #replace infinity values with NaN
X = df.select_dtypes(include='number').drop(columns='y',errors='ignore')
y = df['y']

#SMOTE does not accept missing values encoded as NaN natively.
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)  # X should be numeric

# Apply SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_imputed, y)

print("Before SMOTE:", y.value_counts())
print("After SMOTE:", pd.Series(y_res).value_counts())

##############################################


