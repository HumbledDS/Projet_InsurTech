# %% [markdown]
# # PROJET DATA – DU PYTHON
# 

# %% [markdown]
# ### Plan du Projet DATA :
# 
# _____________________________________________________________
# 
# 1. Traitement de données : compréhension des données, analyse de la 
# complétude et de la conformité des données, contrôles de cohérence
# (univarié/multivarié), gestion des anomalies : mise en place de correctif, 
# exclusions, etc.
# 2. Jointures éventuelles de bases et contrôles 
# 3. Analyses descriptives des données : univariée, multivariée, ASD
# 4. Analyse graphique (data visualisation) + Interfaçage via Shiny for Python 
# 5. Modélisation : supervisée (régression, classification) vs non supervisée ; 
# paramétrique (économétriques) vs non paramétriques (machine learning)
# 6. Analyse des résultats : interprétation, explications 
# 7. Application : prévision, tarification, etc.
# 
# 
# 
# - Understanding the Problem Statement
# - 2. Import Data and Required Packages
#     2.1 Import Packages
#     2.2 Import Data
#     2.3 Show top 5
# - 3. Data Preprocessing 
#     - 3.1 DB_SIN
#         * 3.1.a Data Check and Cleaning (Missing Values, Duplicates, Data Types, Unique values, Anomalies fix)
#         * 3.1.b Preprocessing and Exploratory Analysis
#         * 3.1.c Visualization
#     - 3.2 DB_CNT
#         * 3.2.a Data Check and cleaning (Missing Values, Duplicates, Data Types, Unique values)
#         * 3.2.b Preprocessing and Exploratory Analysis
#         * 3.2.c Visualization
#     - 3.3 DB_TELEMATICS
#         * 3.3.a Data Check and Cleaning (Missing Values, Duplicates, Data Types, Unique values)
#         * 3.3.b Preprocessing and Exploratory Analysis
#         * 3.3.c Visualization
# - 4. DataBase Merges : 
#     * 4.1. Descriptive Data Analysis : Univaried/Multivaried
#     * 4.2. Analyse graphique (data visualisation) + Interfaçage via Shiny for Python 
# - 5. Modélisation
#     * 5.1 supervisée (régression, classification) vs non supervisée ; 
#     * 5.2 paramétrique (économétriques) vs non paramétriques (machine learning)
# - 7. Analyse des résultats : interprétation, explications 
# - 8. Application : prévision, tarification, etc.

# %% [markdown]
# ## 1) Statement
# ##### 1.1 Problem statement
# - Our project revolves around harnessing the potential of recently acquired insurance databases. The objective is to extract meaningful insights and actionable information to enhance our understanding of the insurance landscape.
# - Given the constraints and in line with industry standards, we have chosen Python as our primary tool for data analysis. Python's versatility and extensive libraries, such as Pandas, NumPy, and Scikit-learn, will be crucial in efficiently handling, processing, and analyzing the vast amounts of insurance data at our disposal.
# 
# 
# ##### 1.2 Import Data and Required Packages
# - Dataset Source :
#     * DB_SIN.txt
#     * DB_CNT.xlsx
#     * DB_TELEMATICS.csv
#     

# %% [markdown]
# ## 2. Import Packages
#   Importing Pandas, Numpy, Matplotlib, Seaborn and Warings Library.

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns  # For better aesthetics

# %% [markdown]
# #### 2.2 Import Data

# %%
# Read DB_SIN.txt
sin_df = pd.read_csv('data/DB_SIN.txt', delimiter='\t')  # Assuming it's a tab-separated file

# Read DB_CNT.xlsx from the sheet named 'DB_CNT'
cnt_df = pd.read_excel('data/DB_CNT.xlsx', sheet_name='DB_CNT')

# Read DB_Telematics.csv
telematics_df = pd.read_csv('data/DB_Telematics.csv', delimiter=';')


# %% [markdown]
# #### 2.3 Show top 5

# %%
sin_df.head()

# %% [markdown]
# We immediately see the disparity on the comma placement in AMT_Claim.

# %%
cnt_df.head()

# %% [markdown]
# We notice some matter in Insured.sex inputs, we will investigate it further.

# %%
telematics_df.head()

# %% [markdown]
# ## 3. Data Preprocessing 

# %% [markdown]
# ### 3.1 : DB_Sin

# %% [markdown]
#    - 3.1.a : Data Check and Cleaning (Missing Values, Duplicates, Data Types, Unique values, Anomalies fix)

# %%
sin_df.shape

# %%
sin_df.isna().sum()

# %%
print("DB_SIN.txt DataFrame:")
print(sin_df.info())

# %%
# Check for duplicates in 'Id_pol' and print information for sin_df
duplicates_info_sin = sin_df['Id_pol'].value_counts()
duplicates_info_sin = duplicates_info_sin[duplicates_info_sin > 1]  # Filter only values with more than one occurrence

total_duplicates_sin = duplicates_info_sin.sum()

print("\nDuplicated 'Id_pol' values and their counts in sin_df DataFrame:")
print(duplicates_info_sin)
print(f"\nTotal count of duplicates in sin_df: {total_duplicates_sin}")

# %%
# Check for duplicates in 'Id_pol' in sin_df
duplicates_info_sin = sin_df[sin_df.duplicated(subset=['Id_pol'], keep=False)]

# Filter duplicates where 'AMT_Claim' is equal to 0 and drop those rows
sin_df = sin_df.drop(sin_df[(sin_df['Id_pol'].isin(duplicates_info_sin['Id_pol'])) & (sin_df['AMT_Claim'] == 0)].index)

# Confirm the changes
print("sin_df after removing duplicates with 'AMT_Claim' == 0:")
print(sin_df)

# %%
numeric_summary = sin_df['NB_Claim'].describe()
numeric_summary
unique_values = sin_df['NB_Claim'].unique()

# Display the unique values
print("Unique values in 'NB_Claim':")
print(unique_values)

# %%
# Replace specific values in 'NB_Claim'
sin_df['NB_Claim'] = sin_df['NB_Claim'].replace({'NB_CLAIM:1': '1', 'NB_CLAIM:2': '2'})

unique_values = sin_df['NB_Claim'].unique()
print("Unique values in 'NB_Claim':")
print(unique_values)


# %%
# Display descriptive statistics for numeric columns
numeric_summary = sin_df.describe()

# Display summary for object columns
object_summary = sin_df.describe(include='object')

# Display the results
print("Numeric Summary:")
print(numeric_summary)

print("\nObject Summary:")
print(object_summary)


# %% [markdown]
# We will have to turn NB_Claim and AMT_Claim to integers.

# %%
# Display unique values and their counts in 'AMT_Claim'
unique_values_counts = sin_df['AMT_Claim'].value_counts(dropna=False)

# Display the results
print("Unique values and their counts in 'AMT_Claim':")
print(unique_values_counts)


# %% [markdown]
# Lets fix the ANN situation, we will turn ANN (NaN's) into 0. and convert them to integers since these values are "amounts".
# Secondly, we will take off the decimal part since it'll be easier for operations. AMT_Claim is the Agregated Amount of CLAIM (probably in dollars $, € or £)

# %%
# Convert 'AMT_Claim' to numeric (replace 'ANN' with 0)
sin_df['AMT_Claim'] = sin_df['AMT_Claim'].replace({'ANN': '0'})

# Extract the part before the comma and convert to numeric
sin_df['AMT_Claim'] = sin_df['AMT_Claim'].astype(str).str.split(',').str[0]
sin_df['AMT_Claim'] = pd.to_numeric(sin_df['AMT_Claim'], errors='coerce')

# %%
# Display unique values and their counts in 'AMT_Claim'
unique_values_counts = sin_df['AMT_Claim'].value_counts(dropna=False)

# Display the results
print("Unique values and their counts in 'AMT_Claim':")
print(unique_values_counts)

# %%
# Convert 'NB_Claim' to numeric (replace ',' with '.' if needed)
sin_df['NB_Claim'] = pd.to_numeric(sin_df['NB_Claim'].str.replace(',', '.'), errors='coerce')

# %%
print("DB_SIN.txt DataFrame:")
print(sin_df.info())

# %% [markdown]
# #### Conclusion for Sin_DF data prepocessing : 
# For the SIN_DF data we did the following things to clean up the data :
# 1. NB_Claim : We managed to have 3 differents inputs ('1','2','3') and put them as integers
# 2. AMT_Claim : we managed to put the values as integers and 
# 3. Turned all ANN (NaN's) to 0
# 4. We corrected the 'comma'/'point' situation for the numeric values
# 

# %% [markdown]
# #### 3.1.b Preprocessing and Exploratory Analysis

# %%
sin_df.describe()

# %% [markdown]
# - The mean amount of AMT_Claim is 3136.  
# - The max amount is 104074.  
# - Most of the people (75%) gets above 3702.  
# - 474 (around 10%) did get 0.  
# 
# We can look to see which category of 'NB_Claim' get the most of money (A boxplot with outliers would give the hint).

# %% [markdown]
# #### 3.1.c Visualization
# 
# - 

# %%
sns.boxplot(x='NB_Claim', y='AMT_Claim', data=sin_df, hue='NB_Claim')
plt.title('Boxplot of AMT_Claim by NB_Claim')
plt.show()

# %% [markdown]
# Visualizing the Boxplot of AMT_Claim by NB_Claim we notice the outliers on each category of NB_Claim.  
# While the mean is around 3136 FOR AMT_Claim we observe values going from 10.000 to 104.000 for NB_Claim == 1.  

# %% [markdown]
# ### 3.2 : DB_CNT

# %% [markdown]
#    - 3.2.a : Data Check and Cleaning (Missing Values, Duplicates, Data Types, Unique values, Anomalies fix)

# %%
print("\nDB_CNT.xlsx DataFrame:")
print(cnt_df.info())


# %% [markdown]
# #### The DataSet is shaped as (100399, 12). 100399 rows and 12 columns.  
# - Duration : Duration of the insurance coverage of a given policy, in days  
# - Insured.age : Age of insured driver, in years  
# - Insured.sex : Sex of insured driver (Male/Female)  
# - Car.age : Age of vehicle, in years  
# - Marital : Marital status (Single/Married)  
# - Car.use : Use of vehicle: Private, Commute, Farmer, Commercial  
# - Credit.score : Credit score of insured driver  
# - Region  : Type of region where driver lives: rural, urban  
# - Annual.miles.drive : Annual miles expected to be driven declared by driver  
# - Years.noclaims : Number of years without any claims  
# - Territory : Territorial location of vehicle  

# %% [markdown]
# Conditions to meet :   
# • Duration is the period that policyholder is insured in days, with values in [22,366].  
# • Insured.age is the age of insured driver in integral years, with values in [16,103].  
# • Car.age is the age of vehicle, with values in [-2,20]. Negative values are rare but are possible as buying a newer model can be up to two years in advance.  
# • Years.noclaims is the number of years without any claims, with values in [0, 79] and
# always less than Insured.age.  
# • Territory is the territorial location code of vehicle, which has 55 labels in {11,12,13,· · · ,91}.  

# %%
# Check for duplicates in 'Id_pol' and print information for cnt_df
duplicates_info_cnt = cnt_df['Id_pol'].value_counts()
duplicates_info_cnt = duplicates_info_cnt[duplicates_info_cnt > 1]  # Filter only values with more than one occurrence

total_duplicates_cnt = duplicates_info_cnt.sum()

print("Duplicated 'Id_pol' values and their counts in cnt_df DataFrame:")
print(duplicates_info_cnt)
print(f"\nTotal count of duplicates in cnt_df: {total_duplicates_cnt}")


# %%
# Check for duplicates in 'Id_pol'
duplicates_cnt = cnt_df[cnt_df.duplicated(subset=['Id_pol'], keep=False)]

# Print information about duplicates in cnt_df
print(f"Total count of duplicates in cnt_df: {len(duplicates_cnt)}")



# %%
# Identify and print details of duplicates
duplicates_mask = cnt_df.duplicated(subset=['Id_pol'], keep=False)
duplicates_df = cnt_df[duplicates_mask]

print(f"Total count of duplicates: {len(duplicates_df)}")
print("Details of duplicates:")
print(duplicates_df)

# Remove rows where 'Marital' is empty only for the duplicates
empty_marital_mask = duplicates_df['Marital'].isna() | (duplicates_df['Marital'] == '')
cnt_df = cnt_df[~(duplicates_mask & empty_marital_mask)]

# Check the result
print(f"Total count after removing duplicates with empty 'Marital': {len(cnt_df)}")
cnt_df.dropna(subset=['Marital'], inplace=True)
print(f"We Delete rows where 'Marital' is empty. We notice that most of them are duplicates")


# %%
print("\nDB_CNT.xlsx DataFrame:")
print(cnt_df.info())


# %%
# Filter object variables
object_columns = cnt_df.select_dtypes(include='object').columns

# Display unique values for each object variable
for column in object_columns:
    unique_values = cnt_df[column].unique()
    print(f"Unique values in '{column}':")
    print(unique_values)
    print()

# %% [markdown]
# ##### Imputation of missing Values, and Miss-written Inputs.
# We will proceed as following :
# 
# - Insured.Sex : We will match inputs as Male, Female  
# - Marital : Single, Married  
# - Car use : Private, Commute, Farmer, Commercial. We will use Territory values to commute to the right car use value.  
# - Region : Urban, Rural. With help of terrority we will find the right Region for missing values.  
# - Year.noclaims : EAJ > 0, Then we will turn the column to integer.  

# %%
# Count occurrences of unique values in 'Insured.sex'
sex_counts = cnt_df['Insured.sex'].value_counts()

# Display the result
print("Count of unique values in 'Insured.sex':")
print(sex_counts)
print()

# Count occurrences of unique values in 'Region'
Region_counts = cnt_df['Region'].value_counts()

# Display the result
print("Count of unique values in 'Region':")
print(Region_counts)
print()

# Count occurrences of unique values in 'Marital'
marital_counts = cnt_df['Marital'].value_counts()

# Display the result
print("Count of unique values in 'Marital':")
print(marital_counts)


# %%
# Map values in 'Insured.sex'
sex_mapping = {'Male': ['Male', 'H', 'Unknown'], 'Female': ['Female', 'F']}

# Replace values in 'Insured.sex'
for category, values in sex_mapping.items():
    cnt_df['Insured.sex'] = cnt_df['Insured.sex'].replace(values, category)

# Verify the result
sex_counts_after_mapping = cnt_df['Insured.sex'].value_counts()
print("Count of unique values in 'Insured.sex' after mapping:")
print(sex_counts_after_mapping)

# %%
# Map values in 'Marital'
marital_mapping = {'Single': ['Single', 'Celib'], 'Married': ['Married', 'Marié']}

# Replace values in 'Marital'
for category, values in marital_mapping.items():
    cnt_df['Marital'] = cnt_df['Marital'].replace(values, category)

# Add 'Unknown' and NaN to 'Single'
cnt_df['Marital'] = cnt_df['Marital'].fillna('Single')
cnt_df['Marital'] = cnt_df['Marital'].replace('Unknown', 'Single')

# Verify the result
marital_counts_after_mapping = cnt_df['Marital'].value_counts()
print("Count of unique values in 'Marital' after mapping:")
print(marital_counts_after_mapping)

# %% [markdown]
# • Region  : Type of region where driver lives: rural, urban   
# • Territory is the territorial location code of vehicle, which has 55 labels in {11,12,13,· · · ,91}.  
# To fix the Region values, we will make clusters of Regions.
# The idea here is to see the tuples as following :
# 
# Tuple = [Value(Region) : Territory(1,2,...)].  
# Exemple : [Rural, Territory (1,5,9), Urban, Territory (7,89,63,...)].  
# With this idea, we will then input the approx Region.  

# %%
# Select rows where 'Car.use' is in the specified categories
selected_car_use = ['Private', 'Commute', 'Farmer', 'Commercial']
filtered_df = cnt_df[cnt_df['Car.use'].isin(selected_car_use)]

# Group by 'Car.use' and count the occurrences of each 'Territory'
territory_by_car_use = filtered_df.groupby('Car.use')['Territory'].value_counts()

# Display the result
print(territory_by_car_use)

# %%
# Display unique values and their counts in 'Years.noclaims'
unique_values_counts = cnt_df['Years.noclaims'].value_counts(dropna=False)

# Display the results
print("Unique values and their counts in 'Years.noclaims':")
print(unique_values_counts)

# %%
# Define the allowed car use categories
allowed_car_use = ['Private', 'Commute', 'Farmer', 'Commercial']

# Filter rows where 'Car.use' is not in the allowed categories
invalid_car_use_mask = ~cnt_df['Car.use'].isin(allowed_car_use)

# For invalid 'Car.use', correct based on 'Territory'
for index, row in cnt_df[invalid_car_use_mask].iterrows():
    territory = row['Territory']
    
    # Define mapping from Territory to corrected Car.use
    territory_to_car_use = {
        # Define your mappings here
        1: 'Private',
        2: 'Commute',
        3: 'Farmer',
        4: 'Commercial',
        # Add more mappings as needed
    }

    # Correct the 'Car.use' based on 'Territory'
    corrected_car_use = territory_to_car_use.get(territory, 'Commute')

    # Update the 'Car.use' in the DataFrame
    cnt_df.at[index, 'Car.use'] = corrected_car_use

# Verify the changes
print(cnt_df['Car.use'].value_counts())

# %%
# Define the allowed Region categories
allowed_Region = ['Rural', 'Urban']

# Filter rows where 'Region' is not in the allowed categories
invalid_Region_mask = ~cnt_df['Region'].isin(allowed_Region)

# For invalid 'Region', correct based on 'Territory'
for index, row in cnt_df[invalid_Region_mask].iterrows():
    territory = row['Territory']
    
    # Define mapping from Territory to corrected Region
    territory_to_Region = {
        # Define your mappings here
        1: 'Urban',
        2: 'Rural',
    }

    # Correct the 'Region' based on 'Territory'
    corrected_Region = territory_to_Region.get(territory, 'Urban')

    # Update the 'Region' in the DataFrame
    cnt_df.at[index, 'Region'] = corrected_Region

# Verify the changes
print(cnt_df['Region'].value_counts())

# %%
# Count occurrences of 'EAJ' in 'Years.noclaims'
eaj_count = cnt_df['Years.noclaims'].eq('EAJ').sum()

# Display the result
print("Number of occurrences of 'EAJ':", eaj_count)

# %% [markdown]
# Years.noclaims must be in [0,79].  
# Years.noclaims is the number of years without any claims, with values in [0, 79] and
# always less than Insured.age.
# We will add 'EAJ' to 1 and all the occurances from greater than 79 years will be added back to 79.

# %%
# Replace 'EAJ' with 1
cnt_df['Years.noclaims'] = cnt_df['Years.noclaims'].replace('EAJ', 1)

# Convert 'Years.noclaims' to numeric
cnt_df['Years.noclaims'] = pd.to_numeric(cnt_df['Years.noclaims'], errors='coerce')

# Replace values greater than 79 with 79
cnt_df['Years.noclaims'] = cnt_df['Years.noclaims'].where(cnt_df['Years.noclaims'] <= 79, 79)

# Verify the result
years_noclaims_summary = cnt_df['Years.noclaims'].describe()
print("Summary of 'Years.noclaims' after modifications:")
print(years_noclaims_summary)

# %% [markdown]
# Years.noclaims must always be less than insured.age

# %%
# Check if 'Years.noclaims' is less than 'Insured.age'
invalid_years_noclaims_mask = cnt_df['Years.noclaims'] >= cnt_df['Insured.age']

# Calculate 'Years.noclaims - 16' for rows where the condition is not met
cnt_df.loc[~invalid_years_noclaims_mask, 'Years.noclaims'] = np.maximum(0, cnt_df['Years.noclaims'] - 16)


# %%
# Set values under 22 in 'Duration' to 22
cnt_df.loc[cnt_df['Duration'] < 22, 'Duration'] = 22


# %%
# Filter object variables
object_columns = cnt_df.select_dtypes(include='object').columns

# Display unique values for each object variable
for column in object_columns:
    unique_values = cnt_df[column].unique()
    print(f"Unique values in '{column}':")
    print(unique_values)
    print()

# %% [markdown]
# Conditions Check :   
# • Duration is the period that policyholder is insured in days, with values in [22,366].  
# • Insured.age is the age of insured driver in integral years, with values in [16,103].  
# • Car.age is the age of vehicle, with values in [-2,20]. Negative values are rare but are possible as buying a newer model can be up to two years in advance.  
# • Years.noclaims is the number of years without any claims, with values in [0, 79] and
# always less than Insured.age.  
# • Territory is the territorial location code of vehicle, which has 55 labels in {11,12,13,· · · ,91}.  

# %%
# Find the range of values in 'Duration'
duration_range = (cnt_df['Duration'].min(), cnt_df['Duration'].max())
# Print the range
print("Range of values in 'Duration':", duration_range)


# Find the range of values in 'Insured.Age'
Insured_age_range = (cnt_df['Insured.age'].min(), cnt_df['Insured.age'].max())
# Print the range
print("Range of values in 'Insured.age':", Insured_age_range)


# Find the range of values in 'Car.age'
Car_age_range = (cnt_df['Car.age'].min(), cnt_df['Car.age'].max())
# Print the range
print("Range of values in 'Car.age':", Car_age_range)


# Find the range of values in 'Years.noclaims'
Years_noclaims_range = (cnt_df['Years.noclaims'].min(), cnt_df['Years.noclaims'].max())
# Print the range
print("Range of values in 'Years.noclaims':", Years_noclaims_range)


# Find the range of values in 'Territory'
Territory_range = (cnt_df['Territory'].min(), cnt_df['Territory'].max())
# Print the range
print("Range of values in 'Territory':", Territory_range)

# %% [markdown]
# For simplication purpose, we will turn 'Credit.score' and 'Annual.miles.drive' to integers

# %%
# Convert 'Credit.score' and 'Annual.miles.drive' to integers
cnt_df['Credit.score'] = cnt_df['Credit.score'].astype('int64')
cnt_df['Annual.miles.drive'] = cnt_df['Annual.miles.drive'].astype('int64')

# %%
print("\nDB_CNT.xlsx DataFrame:")
print(cnt_df.info())


# %% [markdown]
# #### 3.2.c Visualization
# 
# - Age and sex
# - Sex and Car use
# - Car Age and Credit Score
# - Car Age and Car use and Region
# - Annual miles and Region
# - Annual miles and Car Use

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for seaborn
sns.set(style="whitegrid")

# Age and Sex
plt.figure(figsize=(10, 6))
sns.boxplot(x='Insured.sex', y='Insured.age', data=cnt_df, palette='pastel')
plt.title('Age Distribution by Sex')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.show()

# Sex and Car use
plt.figure(figsize=(10, 6))
sns.countplot(x='Insured.sex', hue='Car.use', data=cnt_df, palette='pastel')
plt.title('Count of Car Use by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Car Age and Credit Score
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Car.age', y='Credit.score', data=cnt_df, palette='pastel')
plt.title('Scatter Plot of Car Age and Credit Score')
plt.xlabel('Car Age')
plt.ylabel('Credit Score')
plt.show()

# Car Age and Car use and Region
plt.figure(figsize=(12, 6))
sns.boxplot(x='Car.use', y='Car.age', hue='Region', data=cnt_df, palette='pastel')
plt.title('Car Age Distribution by Car Use and Region')
plt.xlabel('Car Use')
plt.ylabel('Car Age')
plt.legend(title='Region')
plt.show()

# Annual miles and Region
plt.figure(figsize=(12, 6))
sns.boxplot(x='Region', y='Annual.miles.drive', data=cnt_df, palette='pastel')
plt.title('Annual Miles Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Annual Miles Driven')
plt.show()

# Annual miles and Car Use
plt.figure(figsize=(12, 6))
sns.boxplot(x='Car.use', y='Annual.miles.drive', data=cnt_df, palette='pastel')
plt.title('Annual Miles Distribution by Car Use')
plt.xlabel('Car Use')
plt.ylabel('Annual Miles Driven')
plt.show()


# %% [markdown]
# #### Commentaire : Conclusion 

# %% [markdown]
# ### 3.3 DB_TELEMATICS
#         

# %%
print("\nDB_Telematics.csv DataFrame:")
print(telematics_df.info())

# %% [markdown]
# * Annual.pct.driven Annualized percentage of time on the road  
# * Total.miles.driven Total distance driven in miles  
# * Pct.drive.xxx Percent of driving day xxx of the week: mon/tue/. . . /sun  
# * Pct.drive.xhrs Percent vehicle driven within x hrs: 2hrs/3hrs/4hrs  
# * Pct.drive.xxx Percent vehicle driven during xxx: wkday/wkend  
# * Pct.drive.rushxx Percent of driving during xx rush hours: am/pm  
# * Avgdays.week Mean number of days used per week  
# * Accel.xxmiles Number of sudden acceleration 6/8/9/. . . /14 mph/s per 1000miles  
# * Brake.xxmiles Number of sudden brakes 6/8/9/. . . /14 mph/s per 1000miles  
# * Left.turn.intensityxx Number of left turn per 1000miles with intensity 08/09/10/11/12  
# * Right.turn.intensityxx Number of right turn per 1000miles with intensity 08/09/10/11/12  

# %% [markdown]
# Conditions to meet for TELEMATICS.csv :  
# • Annual.pct.driven is the number of day a policyholder uses vehicle divided by 365, with
# values in [0,1.1].  
# • Pct.drive.mon, · · · , Pct.drive.sun are compositional variables meaning that the sum
# of seven (days of the week) variables is 100%.  
# • Pct.drive.wkday and Pct.drive.wkend are clearly compositional variables too.  

# %% [markdown]
# DB_CNT has 100399 where DB_TELEMATICS has 100332 lines. 

# %% [markdown]
# #### 3.3.a Data Check and Cleaning (Missing Values, Duplicates, Data Types, Unique values)

# %% [markdown]
# We see that the variable Id_pol is in a different form and some of the columns are objects when they should be floats. Lets start there.

# %%
# Remove 'cnt_' prefix from 'Id_pol'
telematics_df['Id_pol'] = telematics_df['Id_pol'].str.replace('cnt_', '')

# Convert 'Id_pol' to integers
telematics_df['Id_pol'] = telematics_df['Id_pol'].astype('int64')

# Display the updated 'Id_pol' column
print(telematics_df['Id_pol'])


# %%
#Columns to convert to float
columns_to_convert = [
    'Annual.pct.driven', 'Total.miles.driven', 'Pct.drive.mon', 'Pct.drive.tue',
    'Pct.drive.wed', 'Pct.drive.thr', 'Pct.drive.fri', 'Pct.drive.sat', 'Pct.drive.sun',
    'Pct.drive.2hrs', 'Pct.drive.3hrs', 'Pct.drive.4hrs', 'Pct.drive.wkday', 'Pct.drive.wkend',
    'Pct.drive.rush am', 'Pct.drive.rush pm', 'Avgdays.week'
]

columns_to_integer = [
    'Accel.06miles', 'Accel.08miles', 'Accel.09miles', 'Accel.11miles',	'Accel.12miles', 'Accel.14miles',	
    'Brake.06miles', 'Brake.08miles', 'Brake.09miles', 'Brake.11miles',	'Brake.12miles', 'Brake.14miles',	
    'Left.turn.intensity08', 'Left.turn.intensity09', 'Left.turn.intensity10', 'Left.turn.intensity11',	'Left.turn.intensity12',
    'Right.turn.intensity08', 'Right.turn.intensity09', 'Right.turn.intensity10', 'Right.turn.intensity11',	'Right.turn.intensity12'

]

# %%
#We replace ',' to '.'

telematics_df[columns_to_convert] = telematics_df[columns_to_convert].replace(',', '.', regex=True)

# %%
# Convert 'Id_pol' to integers
telematics_df[columns_to_integer] = telematics_df[columns_to_integer].astype('int64')


# %%
telematics_df[columns_to_convert] = telematics_df[columns_to_convert].astype(float)

# %%
# Round float columns to two decimal places
telematics_df[columns_to_convert] = telematics_df[columns_to_convert].round(2)

# %%
missing_values = telematics_df.isnull().sum()
print("Missing Values per Column:")
print(missing_values[missing_values > 0])


# %% [markdown]
# There is no missing Values in the DataSet. 

# %%
print("\nDB_Telematics.csv DataFrame:")
print(telematics_df.info())

# %%
# Check for duplicates in 'Id_pol'
duplicates_db_telematics = telematics_df[telematics_df.duplicated(subset=['Id_pol'], keep=False)]['Id_pol'].unique()
print("Duplicates in DB_Telematics:")
print(duplicates_db_telematics)

# %%
# Find and display duplicated 'Id_pol' values and their counts
duplicates_info = telematics_df['Id_pol'].value_counts()
duplicates_info = duplicates_info[duplicates_info > 1]  # Filter only values with more than one occurrence

total_duplicates = int(duplicates_info.sum()/2)

print("Duplicated 'Id_pol' values and their counts in DB_Telematics DataFrame:")
print(duplicates_info)
print(f"\nTotal count of duplicates: {total_duplicates}")

# %%
# Get the duplicated values in 'Id_pol'
duplicates_db_telematics = telematics_df[telematics_df.duplicated(subset=['Id_pol'], keep=False)]['Id_pol'].unique()

# Remove rows with other variables equal to 0 for duplicated 'Id_pol'
filtered_db_telematics = telematics_df[~((telematics_df['Id_pol'].isin(duplicates_db_telematics)) & (telematics_df.iloc[:, 1:] == 0).any(axis=1))]

print("Filtered DB_Telematics DataFrame:")
print(filtered_db_telematics)

# %%
telematics_df= filtered_db_telematics

# %%
# For telematics_df
#telematics_df.to_csv('artefacts/telematics_df3.csv', index=False)

# %%
# Download the files as csv to into artefact document folder :
# For sin_df
#sin_df.to_csv('artefacts/sin_df2.csv', index=False)

# For cnt_df
#cnt_df.to_csv('artefacts/cnt_df2.csv', index=False)

# For telematics_df
#filtered_db_telematics.to_csv('artefacts/telematics_df2.csv', index=False)

# %% [markdown]
# #### 3.3.b Preprocessing and Exploratory Analysis
#      

# %% [markdown]
# sin_df2
# cnt_df2
# telematics_df2
# 

# %% [markdown]
# #### 3.3.c Visualization

# %% [markdown]
# ## 4. Merging DataBases

# %%
# Merge the datasets on 'Id_pol'
merged_df1 = pd.merge(telematics_df, cnt_df, on='Id_pol', how='inner')

# %%
missing_values = merged_df1.isnull().sum()
print("Missing Values per Column:")
print(missing_values[missing_values > 0])


# %% [markdown]
# ### Feature Engineering : 
# 
# To simplify the dataset and extract meaningful features, we can define the following types of variables through feature engineering:  
# 
# 1. **Demographic Features**:  
#    - **Age Group**: Categorize insured driver's age into groups (e.g., young adult, middle-aged, senior).
#    - **Gender Binary**: Convert 'Insured.sex' into a binary variable (0 for male, 1 for female).  
# 
# 2. **Policy and Vehicle Features**:  
#    - **Policy Duration Category**: Group 'Duration' into categories (e.g., short-term, medium-term, long-term).
#    - **Vehicle Age Group**: Categorize 'Car.age' into groups (e.g., new, moderately old, old).
#    - **Credit Score Group**: Bin 'Credit.score' into categories (e.g., poor, fair, good, excellent).
#    - **Car Use Category**: Create dummy variables for 'Car.use' (Private, Commute, Farmer, Commercial).
#    - **Region Type**: Convert 'Region' into binary variable (0 for rural, 1 for urban).  
# 
# 3. **Driving Behavior Features**:  
#    - **Average Annual Miles Group**: Categorize 'Annual.miles.drive' into groups (e.g., low mileage, moderate mileage, high mileage).
#    - **Years Without Claims Group**: Categorize 'Years.noclaims' into groups (e.g., no claims, 1-2 years, 3-5 years, more than 5 years).
#    - **Territory Code**: Convert 'Territory' into binary variables (one-hot encoding).  
# 
# 4. **Telematics Features**:  
#    - **Driving Intensity**: Sum of 'Left.turn.intensityxx' and 'Right.turn.intensityxx' for overall turning intensity.
#    - **Weekly Driving Patterns**: Sum of 'Pct.drive.xxx' for overall weekly driving patterns.
#    - **Hourly Driving Patterns**: Sum of 'Pct.drive.xhrs' for overall hourly driving patterns.
#    - **Rush Hour Driving**: Sum of 'Pct.drive.rushxx' for overall rush hour driving.
#    - **Safe Driver** : Composite score based on smooth driving behavior with low acceleration, braking, and turn intensity.  
#    - **Aggressive Driver** : Composite score based on high acceleration, braking, and turn intensity.  
# 
# 5. **Response Variables**:  
#    - **Claim Frequency**: Count the number of claims during the observation period ('Response NB Claim').
#    - **Claim Severity**: Aggregate amount of claims during the observation period ('AMT Claim').  
# 
# By engineering these variables, we can simplify the dataset and create features that capture important aspects of driver behavior, vehicle characteristics, and insurance policy details, which can be used for modeling and analysis purposes.  

# %% [markdown]
# Urban > Acceleration > Brake
# 
# Drive.Hour(Rural/urban)
# 
# Correlation : 
# - (Acceleration) > Claims
# - Accelaration > Age/Sex > Rural
# - diff Annual Miles Telematics vs Annual Miles CNT/
# 
# AvgDayWeek
# Drive.rush
# 

# %% [markdown]
# We will merge the bases to be able to work on a larger scale of variables.

# %% [markdown]
# ## 4. DataBase Merges 

# %% [markdown]
# ### 4.1. Descriptive Data Analysis : Univaried/Multivaried

# %% [markdown]
# ### 4.2. Analyse graphique (data visualisation) + Interfaçage via Shiny for Python 

# %% [markdown]
# ## 5. Modélisation

# %% [markdown]
# ### 5.1 supervisée (régression, classification) vs non supervisée    

# %% [markdown]
# ### 5.2 paramétrique (économétriques) vs non paramétriques (machine learning)

# %% [markdown]
# ## 6. Analyse des résultats : interprétation, explications 

# %% [markdown]
# ### 6.1 Interpretation des résultats

# %% [markdown]
# ## 7. Application : prévision, tarification, etc.

# %% [markdown]
# ### 7.1 Prevision and Tarification


