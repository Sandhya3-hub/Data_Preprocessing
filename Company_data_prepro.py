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

#########################################################################          1. BUSINESS UNDERSTANDING
######################################################################

'''
The company wants to identify the key factors that influence its sales performance. By analyzing variables such as price, income, advertising expenditure, and population,the company aims to understand customer demand and develop effective marketing and pricing strategies to increase sales.
'''
#Data Understanding
'''
ales          float64  CompPrice        int64    Income           int64
Advertising      int64  Population       int64    Price            int64
ShelveLoc          str  Age              int64    Education        int64
Urban              str  US                 str
'''
##############################################################
#                1.   EXPLORATORY DATA ANLYSIS
###############################################################
#read the dataset
df=pd.read_csv('Company_Data.csv')
df.head()
print("Shape of Dataset is:",df.shape)
#Shape of Dataset is: (400, 11)

#Check Missing Values
print(df.isnull().sum())
#No NUll values presen in dataset

#Check Datatype of each column
print(df.dtypes)
'''
Sales          float64  CompPrice        int64    Income           int64
Advertising      int64  Population       int64    Price            int64
ShelveLoc          str  Age              int64    Education        int64
Urban              str  US                 str
dtype: object
'''
#Check missing values
print(df.isnull().sum())
#NO  missing values in dataset


#Removing duplicates
df.drop_duplicates(inplace=True)
print("After removing duplicates:",df.shape)  #After removing duplicates: (400, 11)
#No duplicate values in dataset

df.describe().iloc[:,6:12]
'''
          Sales   CompPrice      Income  Advertising  Population       Price
count  400.000000  400.000000  400.000000   400.000000  400.000000  400.000000
mean     7.496325  124.975000   68.657500     6.635000  264.840000  115.795000
std      2.824115   15.334512   27.986037     6.650364  147.376436   23.676664
min      0.000000   77.000000   21.000000     0.000000   10.000000   24.000000
25%      5.390000  115.000000   42.750000     0.000000  139.000000  100.000000
50%      7.490000  125.000000   69.000000     5.000000  272.000000  117.000000
75%      9.320000  135.000000   91.000000    12.000000  398.500000  131.000000
max     16.270000  175.000000  120.000000    29.000000  509.000000  191.000000

             Age   Education
count  400.000000  400.000000
mean    53.322500   13.900000
std     16.200297    2.620528
min     25.000000   10.000000
25%     39.750000   12.000000
50%     54.500000   14.000000
75%     66.000000   16.000000
max     80.000000   18.000000
'''

#inference
''' 

Sales(7.50	7.49	2.82): mean and median of this column is same  but moderate variation in both due to std=2.82 (mean=median)
CompPrice(124.98	125	15.33):Competitor prices average approximately 125, and the mean is nearly equal to the median.(mean=median)
Income:(mean=68.65,median=69,std=27.98):Average income is 68.66, with considerable variation among locations.
Advertising:(6.64	5	6.65)Advertising expenditure varies considerably, and the mean is higher than the median.(right skew)
Population:(264.84	272	147.38)The high standard deviation indicates significant differences in population across the observed locations.
Price:(115.80	117	23.68) Average product price is approximately 115.80, with noticeable variation. there is slight left skew present in feature.(mean<median)
Age:(53.32	54.50	16.20)The average age is approximately 53 years, with a broad age distribution.there is slight left skew present in feature.(mean<median)
Education:(13.90	14	2.62)Average education is approximately 14 years, with relatively low variation compared with the other features.(mean=median)
'''

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
#Varience
''' Variance measures how widely values are spread around their mean.
 A higher variance indicates greater dispersion, while a lower variance 
 indicates that values are closer to their mean. However, 
 variance depends on the feature's measurement scale.'''
'''
 Variance(Second Movement):
 Feature	       Variance	       Inference
 Sales             7.975626      moderate disperssion of data around mean 
CompPrice        235.147243     variability across company price
Income           783.218239     cosiderable disperssion
Advertising       44.227343     vaires across observation
Population     21719.813935     substantial dispersion across different markets.
Price            560.584436     Product prices show considerable variation across                                                              different observations.
Age              262.449618     considerable dispersion around the mean.
Education          6.867168     Education values are comparatively concentrated around              
their mean in absolute terms.

#STD. DEviation:
 Feature	         Variance	      Inference
  Sales            2.824115
  CompPrice       15.334512
  Income          27.986037
  Advertising      6.650364
  Population     147.376436
  Price           23.676664
  Age             16.200297
  Education        2.620528

'''
#Third Movement: Skewness(Symmentry)
skew_val=df.skew(numeric_only=True)
print("\nSkewness(Third Moment)\n:",skew_val)
#inference:
'''
Sales          0.185560      distribution is symmentric with slightly positive skew
CompPrice     -0.042755      distribution is symmentric with negligible negative skew
Income         0.049444      distribution is symmentric with negligible positive skew
Advertising    0.639586      Advertising expenditure is moderately positively skewed,                                     indicating a longer right tail.
Population    -0.051227     distribution is symmentric with negligible negative skew
Price         -0.125286     distribution is symmentric with slight negative skew
Age           -0.077182     distribution is symmentric with slight negative skew
Education      0.044007     distribution is symmentric with negligible positive skew
dtype: float64
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

Kurtosis(Forth Moment):
 Sales         -0.080877  normal distribution and have a lighter tail
CompPrice      0.041666   slightly heavier tail
Income        -1.085289   platykurtic with lighetr tail and flatter distribution
Advertising   -0.545118   platykurtic with lighetr tail and flatter distribution
Population    -1.202318   platykurtic with lighetr tail and flatter distribution
Price          0.451885   The distribution is slightly leptokurtic, indicating somewhat                  heavier tails than a normal distribution.
Age           -1.134392   The distribution is platykurtic, with lighter tails and a flatter shape.
Education     -1.298332   The distribution is platykurtic, with lighter tails and a flatter shape than a normal distribution.
dtype: float64
'''

#Univarent Analysis for numerical features
num_cols = df.select_dtypes(include=['int64','float64']).columns
num_cols
for col in num_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
#inference:
'''
Sales: Sales values are mainly concentrated around 5–9, with the highest frequency near 7–8. The distribution is approximately bell-shaped and shows a slight right-skewed.
  
CompPrice: The data is highly symmetrical with very low negative skewness, meaning prices are evenly balanced above and below the average.
    
Income: Distribution is Bimodel. first high peak ranges from 60 to 70 income bracket, with a frequency exceeding 55. A second, smaller concentration occurs in the 30 to 40 income bracket, peaking at a frequency around 45.after an income of 90, the frequency steadily declines, though there is a minor uptick in the final 110 to 120 bin.

Advertising:  The data is strongly positively skewed (right-skewed). This means that for the vast majority of observations, advertising values are concentrated at the lower end of the scale.
    
Population: The underlying population is uniformly distributed, meaning each value within the 0 to 500 range has a nearly equal probability of occurring.

Price: The data follows an approximately normal (bell-shaped) distribution. It is largely symmetric, meaning most prices cluster around the middle

Age: The data shows a fairly uniform distribution across most age groups, with a mild peak in the late 50s to early 60s and noticeable spikes at both extreme ends (25–30 and 75–80).

Education:The distribution is bimodal, featuring two distinct peaks (modes). This indicates the dataset represents two primary sub-populations with different education levels Lower Peak (11–12 years),Upper Peak (16–17 years).
        
'''
#Count plot for categorical features
cat_cols = df.select_dtypes(include='object').columns
cat_cols
# Index(['ShelveLoc', 'Urban', 'US'], dtype='str')

for col in cat_cols:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col)
        plt.title(f"Count Plot of {col}")
        plt.show()
        
# Compare the bar heights to determine which category is more frequent.
'''
ShelveLoc: 3 categories  medium with high frequency(250) than good(70) and bad(80) category.
Urban: Two categories  Yes with 280  and No with 120 entries
US: Two categories  Yes with 260  and No with 140 entries
'''
    
#Bivariate analysis
#Numerical features vs. target
for col in num_cols:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x='Sales', y=col)
    plt.title(f'{col} vs Sales')
    plt.xlabel('Sales')
    plt.ylabel(col)
    plt.show()
''' 
'''

#Numeric Vs Numeric
plt.figure(figsize=(7, 4))

sns.scatterplot(
    data=df,
    x='Population',
    y='Income',
    hue='Sales'
)

plt.title('Population Vs Sales vs Income')
plt.show()
'''
The points are uniformly and randomly scattered across the plane. This indicates no strong correlation between population size, income level, and sales volume in this specific dataset.

'''
#Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr()
            ,annot=True,cmap='coolwarm')        
plt.title(" features correlation heatmap")
plt.show()
'''
• Sales is most strongly influenced by Price (-0.44) and Advertising (0.27).
• CompPrice and Price share the strongest overall correlation (0.58) in the entire dataset.
• Most demographic variables (Income, Population, Education) show negligible linear relationships with Sales.
Price (-0.44): A moderate negative correlation. As the price of the product increases, Sales tend to decrease significantly. 
• Advertising (0.27): A weak-to-moderate positive correlation. Higher advertising spend is associated with increased sales volume.
• Age (-0.23): A weak negative correlation. This suggests that older customer demographics are associated with slightly lower sales, implying the product might appeal more to a younger audience.
• Income (0.15): A very weak positive correlation. Higher community income levels provide a slight boost to sales, but it is not a primary driver.
• CompPrice vs. Price (0.58): A strong positive correlation. This indicates strategic pricing behavior: when competitors raise or lower their prices, the company adjusts its own price in the same direction.
• Population & Education: Both features show virtually zero meaningful linear correlation with Sales (0.05 and -0.052, respectively) or any other operational variables.
• Advertising vs. Population (0.27): A weak positive relationship, indicating that advertising budgets are slightly larger in areas with higher population densities.

'''
import numpy as np
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

'''    


###################################################################
#                   2. DATA PREPROCESSING
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
    
 #Sales,comprice and pricecolumns have outliers.
 
from feature_engine.outliers import Winsorizer
# For Sales
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['Sales'] 
         )
    
df_t=winsor.fit_transform(df[['Sales']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['Sales'])

#For CompPrice
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['CompPrice'] 
         )
    
df_t=winsor.fit_transform(df[['CompPrice']])
#sns.boxplot(df['CompPrice'])
sns.boxplot(df_t['CompPrice'])

#For Price
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['Price'] #',CompPrice,Price]
         )
    
df_t=winsor.fit_transform(df[['Price']])
#sns.boxplot(df['Price'])
sns.boxplot(df_t['Price'])

###############################################
#Standardization:
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
#skew_val=df.skew(numeric_only=True)
#print("\nSkewness(Third Moment)\n:",skew_val)
for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col],kde=True)
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

#What is a zero-variance feature?
#A feature (column) that has the same value for all rows:
from sklearn.feature_selection import VarianceThreshold

# Create the VarianceThreshold object
selector = VarianceThreshold(threshold=0.0)  # Remove features with 0 variance
# Apply only to numeric features
df_var = selector.fit_transform(df.select_dtypes(include=['float64', 'int64']))

#selector.fit_transform(...) removes any column with variance = 0
########################################################################
#Dummy Variables (One-Hot Encoding)
# Convert categorical variables into dummies
df_dummies = pd.get_dummies(df, drop_first=True)
print("Shape  of orginal dataframe:", df.shape)
print("Shape after dummies:", df_dummies.shape)

#################################################################
#Checking  if is Data Imbalance or not?
# Due to, target variable is numerical no need to use SMOTE for balancing 



