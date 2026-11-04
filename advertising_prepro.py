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
########################################################################
#                   1. BUSINESS UNDERSTANDING
#######################################################################
'''This dataset is used to understand how users demographic characteristics, income daily internet uses,web site activity 
influence their likelihood of clicking on advertisement.
Business can use this information to increase advertisement effect,target \
the right audience and improve advertisement stratergy.
'''
#Data Understanding
'''
Sr.No.  Name of Feature	     Description              Type of Data
1	   Daily_Time_ 
      Spent _on_Site	Average time in minutes
                        a consumer spends on the  
                        website per day.	      Quantitative, Discrete	
                        
2	Age	                Age of the customer	    Quantitative, Continuous	
3	Area_Income	       Average income level
                        of the geographical 
                    area where the consumer 
                                resides.	Quantitative, Discrete	H
4	Daily Internet Usage	Average total minutes per day the consumer spends on the internet. 	Quantitative, Discrete	
5	Ad_Topic_Line	The headline or banner text of the advertisement.	Categoric Nominal 
6	City	The city where the consumer is located.	Categoric Nominal 	Moderatly 
7	Male	Binary indicator for gender (1= Male, 0= Female).	Quantatitive Categorical	
8	Country	The country where the consumer resides.	Categoric Nominal 	Moderatly 
9	Timestamp	Exact date and time when the consumer clicked the ad or closed the browser window	DateTime	To check on what days and time more clicks by users
10	Clicked_on_Ad	The target label for classification.	Quantative Categorical	Wheather user clicked on Ad or Not
		1 = User clicked on the advertisement.		
		0 = User did not click on the advertisement. [1, 2]		

'''
######################################################################
#                2. EXPLORATORY DATA ANALYSIS
####################################################################
#read the dataset
df=pd.read_csv('advertising.csv')
df.head()
print("Shape of Dataset is:",df.shape)
#Shape of Dataset is: (1000, 10)

#Check Missing Values
print(df.isnull().sum())
#No NUll values presen in dataset

#Check Datatype of each column
print(df.dtypes)

'''
Daily_Time_ Spent _on_Site    float64
Age                             int64
Area_Income                   float64
Daily Internet Usage          float64
Ad_Topic_Line                     str
City                              str
Male                            int64
Country                           str
Timestamp                         str
Clicked_on_Ad                   int64
dtype: object
'''

df.describe()
'''
Daily_Time_ Spent _on_Site          Age  ...         Male  Clicked_on_Ad
count                 1000.000000  1000.000000  ...  1000.000000     1000.00000
mean                    65.000200    36.009000  ...     0.481000        0.50000
std                     15.853615     8.785562  ...     0.499889        0.50025
min                     32.600000    19.000000  ...     0.000000        0.00000
25%                     51.360000    29.000000  ...     0.000000        0.00000
50%                     68.215000    35.000000  ...     0.000000        0.50000
75%                     78.547500    42.000000  ...     1.000000        1.00000
max                     91.430000    61.000000  ...     1.000000        1.00000
'''
#Age: Mean is slightly greater than median so slihtly right skewed
#Male column whih is binary where median =0 so no skewness in this column
#clicked on _ad feature have mean=median it mean it has balance classes 
#Check missing values
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

#2. Second Moment: Varience/Standard Daviation(Dispersion)
var_values=df.var(numeric_only=True)
std_values=df.std(numeric_only=True)
print("\n Variance(Second Movement):\n",var_values)
print("\n Standard Daviation(Second Movement):\n",std_values)

#Inference: varience shows  how widely the feature values
#are spread. Age has wide spread than other features  

#Third Movement: Skewness(Symmentry)
skew_val=df.skew(numeric_only=True)
print("\nSkewness(Third Moment)\n:",skew_val)
#inference:
'''
Age (0.479): Shows a mild positive skew-> some users are older than the main concentration of users.
Daily_Time_Spent_on_Site (-0.372): Shows a mild negative skew->some users have relatively low website time.
Daily Internet Usage (-0.034): Is almost symmetric, with very little skewness.
Male (0.076) and Clicked_on_Ad (0.000): Their skewness values are close to zero. 
For binary variables, this depends on the proportions of zeros and ones.
Area_Income: Negative skew. The distribution has a longer tail toward lower income values.
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

Daily Internet Usage (-1.273): Has the lowest kurtosis among the 
continuous features, indicating relatively light tails compared 
with a normal distribution.

Daily_Time_Spent_on_Site (-1.096): Also has a platykurtic distribution.

Age (-0.401): Has slightly lighter tails than a normal distribution.

Area_Income (-0.100): Has kurtosis close to zero, means normally distributed

Male and Clicked_on_Ad: Their kurtosis values are close to -2 
because these are binary features. 
This is expected when their zero and one categories 
are approximately equally represented.
'''
sns.histplot(df.Age,kde=True)  
'''
The histogram shows a roughly bell-shaped curve(unimodel)
somewhat close to a normal distribution,    
but slightly rigth-skewed(longer tail on the right)
This means that there some older users than the  main constration of users.
'''
#sns.histplot(df.Daily Internet Usage,kde=True)  
sns.histplot(df.Area_Income,kde=True)  
'''
The histogram shows a roughly bell-shaped curve(unimodel)
but slightly left-skewed(longer tail on the left)
The distribution has a longer tail toward lower income values. 
Means most of user have larger income '''

sns.histplot(df.Male,kde=True)  
'''
Binary Feature so have balance values

'''

#Scatter plot:Bivariate analysis

#Analyze the relationship between two features to understand how they are associated.
sns.scatterplot(data=df,x='Age',y='Area_Income',hue='Clicked_on_Ad')
plt.title('Scatter Plot: Age Vs Area_Income')
plt.show() 
'''
The plot indicates that older users tend to have a higher advertisement 
click rate than younger users. 
Additionally, users with higher area incomes appear less likely to 
click on advertisements compared to users with lower area incomes.
'''
sns.scatterplot(data=df,x='Age',y='Male',hue='Clicked_on_Ad')
plt.title('Scatter Plot: Age Vs Gender')
plt.show() 
'''
plot shows that female older users tends to have a highter advertisement 
clicks than younger ones.
also it shows older male users tends to have a highter advertisement 
clicks than younger ones,but not more than female user.
'''

#Box Plot 

# Select numerical features
features = [
    'Age',
    'Area_Income',
    'Daily_Time_ Spent _on_Site',
    'Daily Internet Usage'
]

# Create separate box plots
for feature in features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.xlabel(feature)
    plt.show()
'''
->Age feature has no outlier and the middle 50% of the values, 
between the first quartile (Q1) and third quartile (Q3) 
that is in range of age 28 to 43
-> Area_income feature:
    Middle 50%: The box extends approximately from ₹47,000 (Q1) 
    to ₹67,500 (Q3). 
    This means half of the users' area income values fall 
    within this range.
Whiskers: The lower whisker extends to around ₹20,000 
and the upper whisker to around ₹79,500, 
showing the range of non-outlier values.
Potential outliers: Several points appear below ₹20,000, 
indicating unusually low area income values compared 
with the rest of the dataset.'''

sns.boxplot(data=df,x='Clicked_on_Ad', y='Area_Income')

plt.title('Area Income by Ad Click')
plt.xlabel('Clicked on Ad (0 = No, 1 = Yes)')
plt.ylabel('Area Income')
plt.show()
'''
Users who clicked on the advertisement generally belong 
to lower-income areas, whereas non-clicking users tend to 
belong to higher-income areas.
non-clicked users has income median near aout 63000 
and clicked user income median is about 48000.
so h\as the income rises may be clocked on AD may get decrease. 
so it shows negative relationship.
'''
sns.boxplot(data=df,x='Clicked_on_Ad', y='Daily_Time_ Spent _on_Site')

plt.title('Daily time spent on internet by Ad Click')
plt.xlabel('Clicked on Ad (0 = No, 1 = Yes)')
plt.ylabel('Daily_Time_Spent_on_Site')
plt.show()
'''
Both class have outliers which shows some users are unusal to spent time on internet and clicked on ad.
non-clicked group(0) spend more time on internet than clicked group(1). 
median of non-clicked group is about 78 mins as compere to median of clicked group (55 min).
clicked user has wide spread values than non-clicked group,
indicating greater variation in daily time spend on internet than the non-clicked group.
'''
sns.boxplot(data=df,x='Clicked_on_Ad', y='Daily Internet Usage')
plt.title('Daily Internet Usage by Ad Click')
plt.xlabel('Clicked on Ad (0 = No, 1 = Yes)')
plt.ylabel('Daily Internet Usage')
plt.show()
'''
class 0(non-clicked group) has ouliers representing users whose 
internet usage differs considerably from the typical values in their respective group.
non cliked group uses more internet than clicked group. 
clicked group has wide spread values than non-clicked group,
indicating greater variation in daily internet usage than the non-clicked group.
'''
#Joint plot:
sns.jointplot(data=df,x='Age',y='Daily Internet Usage',hue='Clicked_on_Ad')
plt.suptitle("Joint plt with Clicked on AD Hue")
plt.show()

''' 
Middle age group with less internet usage are tends to clicked group and 
daily internet usage is slightly right skewed means 
very less users spend more time on the internet
'''

#Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True)
            ,annot=True,cmap='coolwarm')        
plt.title(" features correlation heatmap")
plt.show()

'''
heatmap shows the relationship between different features in the Advertisement dataset,
 especially how they are related to the target feature Clicked_on_Ad.
 Inference of the correlation heatmap:
1. Daily Time Spent on Site (-0.75): It has a strong negative correlation with Clicked_on_Ad. 
As time spent on the site increases, ad clicks tend to decrease.

2. Age (0.49): It has a moderate positive correlation with Clicked_on_Ad. 
Older users tend to be more likely to click on advertisements.

3. Area Income (-0.48): It has a moderate negative correlation with Clicked_on_Ad. 
Users with higher area income tend to be less likely to click on advertisements.

4. Daily Internet Usage (-0.79): It has a strong negative correlation with Clicked_on_Ad. 
Higher daily internet usage is associated with a lower likelihood of clicking on advertisements.

5. Male (-0.038): It has a very weak negative correlation with Clicked_on_Ad, 
indicating almost no linear relationship between gender and ad clicks.
'''

#CDF and PDF

#num_cols=df.select_dtypes(include=np.number).columns.drop('id')
#num_cols = df.select_dtypes(include=np.number).columns.drop('id')
num_cols = df.select_dtypes(include=np.number)
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
PDF:
1. The PDF plot of Clicked_on_Ad displays a bimodal distribution with peaks around 0 and 1, 
reflecting the two categories of the binary target variable.

2.  The pdf of daily time spent feature  shows bimodel distribution with peaks around 43 and 80,
 reflecting two distinct groupt spending time. 
 
 Low engagement group(40-43min):representing a cohort of users who spend a relatively shorter amount 
 of time on the site.
  
 High engagement Group(80-83 min):representing a larger cohort of highly engaged users who spend a significant amount of time on the site.
 
 CDF:around 90 units of time spent by almost all users(100%).
 65 mins time spent by 50% users.
 
 
3. Age:it is a unimodal distribution with a right-skewed (positively skewed) tail, 
indicating that more individuals are concentrated in younger-to-middle adulthood while 
fewer individuals extend into older age brackets.

CDF: • 
	• Under 30: Approximately 35% of the population is 30 years old or younger.
	• Under 40: Roughly 70% of the population is 40 years old or younger, meaning only about 30% of the individuals are older than 40.
	• Under 60: Close to 100% of the population is under 60 years old.

       
4. Income:this shows a negatively left skwed distribution.This means a large portion of the population is 
concentrated at higher income levels(60000-65000),with a longer tail extending toward the lower income levels (on the left). 
The vast majority of the data lies between 45,000 and 75,000, where the curve is tallest and widest.

CDF: 
    • The 50th percentile (Cumulative Probability = 0.5) lies at approximately $55,000. This means half of the areas earn less than $55,000, and half earn more.
    • Approximately 80% of the data points fall below $65,000. 
    Consequently, only the top 20% highest-earning areas have an income between $65,000 and $80,000+.
    
5. daily _internet usage : bimodel distribution with peaks around 120 and 220. 
lower intensity group(120): representing large group of lower intensity internet users.
higher intensity group(220): representing group with higher intensity internet users.

CDF: 
    caps off around 265 units (100% cumulative probability).
    it aligns with roughly 180–190 units.(50% Cummulative probability)
    
    
'''    


####################################################################
#          3. DATA PREPROCESSING
###################################################################

#Seperate  the numerical and categorical columns
cat_cols = df.select_dtypes(include='object').columns
cat_cols
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_cols

#Removing duplicates
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)
#No duplicate values present in Dataset

#Outlier Treatment
for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.xlabel(col)
    plt.show()
    
#Area_income has outlier
 
from feature_engine.outliers import Winsorizer
# For 'Area_income'
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['Area_Income'] #',CompPrice,Price]
         )
    
df_t=winsor.fit_transform(df[['Area_Income']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['Area_Income'])

###############################################
#Standardization:
#Standardization is generally a good choice for scaling numerical features, especially if you plan to use machine learning algorithms such as Logistic Regression, SVM, KNN, or PCA.
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

# Apply only to numeric features
df_var = selector.fit_transform(df.select_dtypes(include=['float64', 'int64']))
#selector.fit_transform(...) removes any column with variance = 0
print("Shape  of orginal dataframe:", df_var.shape)
#(1000,6)
#No columns with Zeo Variance
########################################################################
#Dummy Variables (One-Hot Encoding)
# Convert categorical variables into dummies
df_dummies = pd.get_dummies(df, 
                            columns=['City', 'Country'],
                            dtype=int,
                            drop_first=True)
print("Shape  of orginal dataframe:", df.shape)
print("Shape after dummies:", df_dummies.shape)
#no need to do one hot encodeing as 
#################################################################

df['Clicked_on_Ad'].value_counts()
#Clicked_on_Ad
#0    500
#1    500
#No data imbalance So no need of SMOTE 
###########################################################################



