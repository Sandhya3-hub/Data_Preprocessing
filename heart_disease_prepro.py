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
'''Healthcare organizations want to identify the important factors associated with heart disease so that patients at higher risk can be identified earlier. 
By analyzing patient information such as age, sex, chest pain type, blood pressure, cholesterol, 
maximum heart rate, and other clinical indicators, the dataset can help explore patterns associated
with heart disease and support clinical decision-making.
'''
#Data Understanding
'''
age       int64     sex       int64  cp       int64  trestbps      int64
chol      int64    fbs       int64  restecg   int64  thalach       int64
exang     int64   oldpeak    float64  slope   int64  ca            int64
thal      int64   target      int64

'''

########################################################################
#                  2.  EXPLORATORY DATA ANLYSIS
#######################################################################
#1. Data Understanding

#read the dataset
df=pd.read_csv('heart_disease.csv')
df.head()
print("Shape of Dataset is:",df.shape)
#Shape of Dataset is: (303,14)

#Check Missing Values
print(df.isnull().sum())
#No NUll values presen in dataset

#Check Datatype of each column
print(df.dtypes)
'''
age       int64     sex        int64  cp        int64  trestbps      int64
chol      int64    fbs         int64  restecg   int64  thalach       int64
exang     int64   oldpeak    float64  slope     int64  ca            int64
thal      int64   target      int64
dtype: object

'''

df.describe().iloc[:,0:6]
'''
              age         sex          cp    trestbps        chol         fbs
count  303.000000  303.000000  303.000000  303.000000  303.000000  303.000000
mean    54.366337    0.683168    0.966997  131.623762  246.264026    0.148515
std      9.082101    0.466011    1.032052   17.538143   51.830751    0.356198
min     29.000000    0.000000    0.000000   94.000000  126.000000    0.000000
25%     47.500000    0.000000    0.000000  120.000000  211.000000    0.000000
50%     55.000000    1.000000    1.000000  130.000000  240.000000    0.000000
75%     61.000000    1.000000    2.000000  140.000000  274.500000    0.000000
max     77.000000    1.000000    3.000000  200.000000  564.000000    1.000000
age: The average age is 54.37 years, and the median is 55 years.
    The mean and median are close, suggesting that the age distribution may be relatively balanced
    The middle 50% of observations fall between 47.5 and 61 years.
Sex:
    The mean is 0.683, indicating that approximately 68.3% of observations belong to category 1.
cp: The mean chest pain code is 0.967, and the median is 1.Values range from 0 to 3, with a 
    standard deviation of 1.03.
    The middle 50% of observations have codes between 0 and 2.
Resting Bloood Prssure:
    The average resting blood pressure is 131.62 mm Hg, while the median is 130 mm Hg.
    Values range from 94 to 200 mm Hg, with a standard deviation of 17.54 mm Hg.
chol: mean is 246 and median is 240. where mean >median so slitly right skewed.
      50% data poits fall between 211 to 274 mg/dL
Fasting BS(fbs):
          The mean is 0.149, indicating that approximately 14.9% of observations belong to category 1.
          Approximately 85.1% belong to category 0.
          The median and 75th percentile are both 0, showing that category 0 is predominant.
'''
df.describe().iloc[:,6:12]
df.describe().iloc[:,12:14]

#
'''
         restecg     thalach       exang     oldpeak       slope          ca
count  303.000000  303.000000  303.000000  303.000000  303.000000  303.000000
mean     0.528053  149.646865    0.326733    1.039604    1.399340    0.729373
std      0.525860   22.905161    0.469794    1.161075    0.616226    1.022606
min      0.000000   71.000000    0.000000    0.000000    0.000000    0.000000
25%      0.000000  133.500000    0.000000    0.000000    1.000000    0.000000
50%      1.000000  153.000000    0.000000    0.800000    1.000000    0.000000
75%      1.000000  166.000000    1.000000    1.600000    2.000000    1.000000
max      2.000000  202.000000    1.000000    6.200000    2.000000    4.000000

          thal      target
count  303.000000  303.000000
mean     2.313531    0.544554
std      0.612277    0.498835
min      0.000000    0.000000
25%      2.000000    0.000000
50%      2.000000    1.000000
75%      3.000000    1.000000
max      3.000000    1.000000'''
#inference
'''
thalach:The mean maximum heart rate is 149.65, with a median of 153. 
Values range from 71 to 202. The middle 50% of observations
lie between 133.5 and 166.
oldpeak:The mean is 1.04 and the median is 0.8. 
Values range from 0 to 6.2. The mean being greater than the median 
suggests a right-skewed distribution, with some relatively high values.
target: The mean is 0.545. Assuming 1 indicates the presence of heart 
disease and 0 indicates its absence, approximately 54.5% of the records 
belong to the heart-disease-positive class, while 45.5% belong to the negative class.
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
 Feature	Variance	Inference
age	        82.48	     Shows noticeable variation in the ages of 
                         individuals in the dataset.
trestbps	307.59	    Shows considerable variation in resting 
                        blood pressure among individuals.
chol	    2686.43	    Has high numerical variance, indicating a wide 
                        spread in serum cholesterol levels.
fbs     	0.127	    Low variance because fasting blood sugar is 
                        binary-coded. Most observations belong to one category.
thalach	    524.65	     Indicates considerable variation in maximum heart rate achieved.                     
#STD. DEviation:
Highest standard deviation: chol (51.83), followed by thalach (22.91) 
and trestbps (17.54). These features have the greatest numerical spread,
although their measurement units differ.
Lowest standard deviation: fbs (0.356), among the listed features. 
Because it is binary-coded, its standard deviation reflects the 
proportions of the two categories.
Target: The standard deviation of 0.499 is close to the maximum of 0.5 
for a binary variable, indicating that the two target classes are 
relatively balanced.
'''
#Third Movement: Skewness(Symmentry)
skew_val=df.skew(numeric_only=True)
print("\nSkewness(Third Moment)\n:",skew_val)
#inference:
'''
Highest positive skewness: fbs (1.987), followed by ca (1.310), 
oldpeak (1.270), and chol (1.143).
Highest negative skewness: sex (-0.791), followed by thalach (-0.537) 
and slope (-0.508).
Approximately symmetric features: age (-0.202), restecg (0.163), 
and target (-0.180), although category coding affects interpretation for categorical variables.
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

Highest kurtosis: chol (4.505), suggesting heavy tails and possible
extreme cholesterol values.
Other positively kurtotic numerical features: oldpeak (1.576), trestbps (0.929), and ca (0.839).
Kurtosis close to zero: thalach (-0.062), whose tail weight is relatively 
close to that of a normal distribution.
Most negative kurtosis: target (-1.981), which is expected for a binary
 variable with relatively balanced classes.
'''

#Univarent Analysis for numerical features
num_cols = [
    'age', 'trestbps', 'chol',
    'thalach', 'oldpeak','target'
]

for col in num_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
#inference:
'''
age:
   shows bimodel where one paek at age 42 and second at 61. 
   symmetric distribution with a slight left skew.
trestbps:The distribution is moderately right-skewed, suggesting a longer
    tail toward higher blood pressure values. The histogram can help identify where observations are concentrated and whether unusually high values occur.
chol:The distribution is positively skewed, with a longer right tail. 
    The high kurtosis suggests possible extreme observations.
thalach: The distribution is slightly left-skewed, indicating a longer 
tail toward lower heart-rate values. Its kurtosis is close to zero.
oldpeak:The distribution is positively skewed, with many observations 
potentially concentrated at lower values and a tail extending 
toward higher values.
'''
#Count plot for categorical features
cat_cols = [
    'sex', 'cp', 'fbs', 'restecg',
    'exang', 'slope', 'ca', 'thal'
]

for col in cat_cols:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col)
        plt.title(f"Count Plot of {col}")
        plt.show()
# Compare the bar heights to determine which category is more frequent.
'''
sex:Shows the distribution of the two encoded sex categories. 
0 is less than 1
cp:Shows the frequency of each chest pain category.
0 category is most frequent while 3rd is least frequent
fbs: Displays the frequency of the two fasting blood sugar categories.
Suggests that category 1 represents a relatively small proportion of observations.
restecg:Shows the distribution of resting ECG categories.
categor1 is dominant  foolwed by category 2.
exang:Displays the frequency of exercise-induced angina categories.
   mean of 0.327 indicates that approximately 32.7% of records have
   exang = 1.
slope: Shows the frequency of the different ST-segment slope categories.
category 2 slightly higher than category 1.
ca:Displays the distribution of the number of major vessels recorded.
median of 0 indicates that at least half of observations have ca = 0.
thal:Shows the frequency of the coded thalassemia-related categories.
category 2 has highest bar height than other 3 categories.
target:Shows the distribution of heart disease outcome classes.
 mean of 0.545 indicates approximately 54.5% target-positive records
 and 45.5% target-negative records, 
 assuming 1 means heart disease present.
'''
    
#Bivariate analysis
#Numerical features vs. target
for col in num_cols:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x='target', y=col)
    plt.title(f'{col} vs Target')
    plt.xlabel('Target (0 = No, 1 = Yes)')
    plt.ylabel(col)
    plt.show()
'''  Age: 
    The median age of  group 0 is higher(roughly 58 years)
    Group 1  median age is lower, at approximately 52 years old.
    so People in Group 1 (Target = Yes) tends to be younger on average 
    than those in Group 0.
    spread of group 0 is from 52 to 62 and group 1 has 42 to 58. so group 1 has wide spread(between 44 to 52)
    There is a single outlier in group 0 at around 35 years old, 
    indicating an unusually young individual for this group.
    the two groups heavily overlap so age feature alone  may not distinguish the classes well.
trestbps:median of both group is same. group 0 has wide spread than group 1.
         outlier present in both groups at right side(high ends) group 0  values ranges 
         from 100 to 180 where 2 data points beyond the 180.  Group 1 ranges from 30 to 170 where 3 datapoints beyond 170.
         blood pressure alone may not distinguish the classes well.
chol: median of group 0 is slightly higher, at around 250 mg/dL
     than group 1 which sits around 235 mg/dL.
     Both groups show a very similar spread in their middle 50% of data. 
     Group 0 ranges roughly from 220 to 280 mg/dL, while Group 1 ranges 
     from about 210 to 270 mg/dL.
     Group 1 (Yes) has a much higher number of extreme high-cholesterol outliers, 
     with one individual peaking near 560 mg/dL. 
     Group 0 (No) has only one distinct outlier around 410 mg/dL.
    the two groups heavily overlap. This suggests that cholesterol level by 
    itself may not be a strong standalone differentiator for predicting the target variable in this specific dataset.
thalach: 
	for Target = 0, the median maximum heart rate sits around 142 bpm.
	For Target = 1, the median maximum heart rate is noticeably higher, around 161 bpm.
    The middle 50% of patients with Target = 0) fall between roughly 125 and 156 bpm.
    The middle 50% of patients with Target = 1) are shifted upward, 
    falling between roughly 149 and 172 bpm.
    Target = 0 has a single low outlier near 71 bpm.
	Target = 1 has a cluster of lower outliers ranging between 95 and 115 bpm.
   higher maximum heart rates (thalach) are associated with a Target value of 1.
oldpeak: • 
	 for target 0: the median is noticeably higher, approximately 1.4.
	• For Target 1 (Yes), the median is much lower, at roughly 0.2.
    Target = 0 group shows greater variability. The middle 50% of the data 
    spans from about 0.6 to 2.5.
    The Target = 1 group is heavily concentrated at lower values. 
    The box spans from 0 to 1.0, indicating that at least 25% of the individuals 
    in this group have an oldpeak value of exactly 0.
    Target = 0 has two extreme outliers near 5.6 and 6.2.
    Target = 1 has several outliers trailing upwards at around 2.6, 3.0, 3.5,
    and 4.2.
    This distribution suggests that oldpeak could be a strong predictor or feature for distinguishing between these two target groups.
'''
#categorical Vs target
for col in cat_cols:
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x=col, hue='target')
    plt.title(f'{col} vs Target')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()
#Joint plot:
sns.jointplot(data=df,x='Age',y='Daily Internet Usage',hue='Clicked_on_Ad')
plt.suptitle("Joint plt with Clicked on AD Hue")
plt.show()

for col in cat_cols:
    print(f"\n{col} vs target (%):")
    print(
        pd.crosstab(
            df[col],
            df['target'],
            normalize='index'
        ).mul(100).round(2)
    )
# Percentage of target 0 and target 1 for each category.
'''
sex vs target (%):	
target      0      1
sex                 
0       25.00  75.00
1       55.07  44.93

cp vs target (%):	
target      0      1
cp                  
0       72.73  27.27
1       18.00  82.00
2       20.69  79.31
3       30.43  69.57

fbs vs target (%):	
target      0      1
fbs                 
0       44.96  55.04
1       48.89  51.11

restecg vs target (%):	
target       0      1
restecg              
0        53.74  46.26
1        36.84  63.16
2        75.00  25.00

exang vs target (%):	
target      0      1
exang               
0       30.39  69.61
1       76.77  23.23

slope vs target (%):	
target      0      1
slope               
0       57.14  42.86
1       65.00  35.00
2       24.65  75.35

ca vs target (%):	
target      0      1
ca                  
0       25.71  74.29
1       67.69  32.31
2       81.58  18.42
3       85.00  15.00
4       20.00  80.00

thal vs target (%):	
target      0      1
thal                
0       50.00  50.00
1       66.67  33.33
2       21.69  78.31
3       76.07  23.93
'''
#Numeric Vs Numeric
plt.figure(figsize=(7, 4))

sns.scatterplot(
    data=df,
    x='age',
    y='thalach',
    hue='target'
)

plt.title('Age vs Maximum Heart Rate')
plt.show()
'''
As age increases, the maximum heart rate (thalach) generally decreases.
target 1 clustered at the top left, showing that individuals with higher maximum heart rates 
relative to younger age group.target 0 represent individuals without heart disease, clustered noticeably towards the lower and older regions of the plot.

'''
#Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr()
            ,annot=True,cmap='coolwarm')        
plt.title(" features correlation heatmap")
plt.show()
'''
1.  Red shades (Positive numbers up to 1.0): Indicate a positive correlation. As one variable goes up, the other tends to go up.
Light Blue/Grey shades (Around 0.0): Indicate weak or no linear correlation. The variables do not strongly influence each other linearly.
Dark Blue shades (Negative numbers down to -0.4): Indicate a negative correlation. As one variable goes up, the other tends to go down.
2.
	• thalach (0.42): This has the strongest positive correlation with the target. As maximum heart rate increases, the value of the target variable is more likely to increase.
	• oldpeak (-0.43): This has the strongest negative correlation with the target. As ST depression decreases, the target value is more likely to increase (and vice versa).
3. age and thalach (-0.4): There is a noticeable negative correlation here, confirming the well-known 
medical fact that maximum heart rate (thalach) generally decreases as a person's age increases.
4. 
	• chol and thalach (-0.0099): The value is nearly zero, meaning there is virtually no linear relationship 
    between a person's cholesterol levels and their maximum achieved heart rate in this dataset.

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

'''    
######################################################################
##                3.  DATA PREPROCESSING
#####################################################################
#Seperate  the numerical and categorical columns
cat_cols = df.select_dtypes(include='object').columns
cat_cols
#No categorical columns presen in this dataset
num_cols = df.select_dtypes(include=['int64','float64']).columns
num_cols

#Removing duplicates
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)  #(302,14)
#One record found as duplicate in Dataset

#Outlier Treatment
for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.xlabel(col)
    plt.show()
    
#thal,ca,trestbps,chol,thalach,oldpeak
 
from feature_engine.outliers import Winsorizer
# For 'thal'
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['thal']
         )
    
df_t=winsor.fit_transform(df[['thal']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['thal'])

#ca
winsor= Winsorizer(
           capping_method= 'iqr',
           tail='both',
           variables=['ca']
           )
      
df_t=winsor.fit_transform(df[['ca']])
  #sns.boxplot(df['Sales'])
sns.boxplot(df_t['ca'])

#sns.boxplot(df_t)

# trestbps,chol,thalach,oldpeak
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         variables=['trestbps']
         )
    
df_t=winsor.fit_transform(df[['trestbps']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['trestbps'])

#chol
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['chol']
         )
    
df_t=winsor.fit_transform(df[['chol']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['chol'])

#thalach
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['thalach']
         )
    
df_t=winsor.fit_transform(df[['thalach']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['thalach'])

#oldpeak
winsor= Winsorizer(
         capping_method= 'iqr',
         tail='both',
         fold=1.5,
         variables=['oldpeak']
         )
    
df_t=winsor.fit_transform(df[['oldpeak']])
#sns.boxplot(df['Sales'])
sns.boxplot(df_t['oldpeak'])
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
    #sns.histplot(df[col], kde=True)\
    #.set_title("Final Transformed")
    plt.show()
    #fbs,sex, exang,
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

df_var = selector.fit_transform(df[num_cols])

#selector.fit_transform(...) removes any column with variance = 0
print("Shape  of orginal dataframe:", df_var.shape)
#(302,14)
#No columns with Zeo Variance
########################################################################
#there is no categorical feature in dataset so no need of one hot encoding
#################################################################
#Checking  if is Data Imbalance or not?
# 1. Check for highly skewed features
#new_num_cols = [col for col in num_cols if col != 'pdays']
skewness = df[num_cols].skew()
high_skew = skewness[abs(skewness) > 3] # Common threshold for extreme skew
print("Highly skewed features:\n", high_skew)
# No columns are highly skewed

# 2. Check for zero-inflation or single-value dominance
for col in df.columns:
    top_value_pct = df[col].value_counts(normalize=True).iloc[0]
    print(top_value_pct)
    if top_value_pct > 0.85: # If one value dominates > 85% of the data
        print(f"Feature '{col}' is heavily imbalanced. Top value occupies {top_value_pct*100:.2f}% of data.")
#Feature 'fbs' is heavily imbalanced. Top value occupies 85.10% of data.
#Need to balance imbalance dataset So need to use SMOTE 
###########################################################################
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
#SMOTE
# Split features and target
df['target'] = df['target'].astype(int)
df['target'].value_counts()
df['fbs'].value_counts()
X=df.select_dtypes(include='number').drop(columns='target',errors='ignore')
y = df['target']

#SMOTE does not accept missing values encoded as NaN natively.
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)  # X should be numeric

# Apply SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_imputed, y)

print("Before SMOTE:", y.value_counts())
print("After SMOTE:", pd.Series(y_res).value_counts())

##############################################


