#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[2]:


import pandas as pd

# Load data from a CSV file
df = pd.read_csv('/Users/HP/OneDrive/Desktop/stock_raw_data.csv')
print(df.head())


# In[3]:


pip install pandas


# In[5]:


import pandas as pd

df = pd.read_csv('/Users/HP/OneDrive/Desktop/stock_raw_data.csv')
df
# print(df.head())


# In[6]:


# 3. Explore the Data
# Inspect the first few rows, column names, and basic statistics:


print(df.head())         # First few rows
print(df.info())         # Summary of DataFrame including column names, types, and non-null counts
print(df.describe())    # Basic statistics for numerical columns


# In[7]:


# 4. Handle Missing Data
# Detect missing values:


print(df.isnull().sum())  # Count of missing values per column


# In[8]:


# Fill missing values:
import pandas as pd

df = pd.read_csv('/Users/HP/OneDrive/Desktop/stock_raw_data.csv')

# Fill with a specific value
df.fillna(value=0, inplace=True)

print("\nDataFrame after filling missing values with 0:")
print(df)



# In[15]:


import pandas as pd
df = pd.read_csv('/Users/HP/OneDrive/Desktop/stock_raw_data.csv')

df


# In[6]:


# mean of high price

mean_value=df['high'].mean()
print('Mean Value of high : '+str(mean_value))

# mean of low price

mean_value=df['low'].mean()
print('Mean Value of low: '+str(mean_value))


# In[8]:


# median of high price

mean_value=df['high'].median()
print('median Value of high: '+str(mean_value))

# median of low price

mean_value=df['low'].median()
print('median Value of low: '+str(mean_value))


# In[10]:


# Drop rows with any missing values
df.dropna(inplace=True)

print("\nDataFrame after dropping rows with any missing values:")
print(df)


# In[16]:


# Drop missing values:

df.dropna(subset=['high', 'low','open','close'], inplace=True)
print(df)


# In[17]:


# 5. Remove Duplicates

df.drop_duplicates(inplace=True)
print(df)


# In[26]:


pip install --upgrade pip


# In[27]:


pip install scikit-learn


# In[4]:


# 6. Normalize/Standardize Data
# For numerical features, you might want to normalize or standardize:

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('/Users/HP/OneDrive/Desktop/stock_raw_data.csv')
scaler = MinMaxScaler()

# Fit and transform the data
df[['high', 'low']] = scaler.fit_transform(df[['high', 'low']])

print(df)


# In[14]:


from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('/Users/HP/OneDrive/Desktop/stock_raw_data.csv')

# Convert date strings to datetime objects
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

# Extract useful numeric features from the datetime
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Drop the original date column if no longer needed
df = df.drop(columns=['date'])

print(df)

scaler = StandardScaler()

df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(df_standardized)



# In[34]:


# 7. Handle Outliers
# Identify and handle outliers, which might involve removing or transforming them:

import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame
df = pd.read_csv('/Users/HP/OneDrive/Desktop/stock_raw_data.csv')

# # Box plot
# plt.boxplot(df['high'])
# plt.boxplot(df['low'])
# plt.title('Share price')
# plt.show()

# Scatter plot
plt.scatter(df['open'], df['close'])
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')

# ffrom scipy import stats

# # Calculate Z-scores
# df['z_score'] = stats.zscore(df['values'])
# print(df)

# # Identify outliers based on Z-score
# outliers = df[abs(df['z_score']) > 3]
# print("Outliers:", outliers)


# # Identify outliers based on Z-score
# outliers = df[abs(df['z_score']) > 3]
# print("Outliers:", outliers)

# Calculate IQR
Q1 = df['high'].quantile(0.25)
Q3 = df['low'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['high'] < lower_bound) | (df['low'] > upper_bound)]
print("Outliers:", outliers)


# plt.show()


# In[27]:


# Sample DataFrame
df = pd.read_csv('/Users/HP/OneDrive/Desktop/stock_raw_data.csv')

# Scatter plot
plt.scatter(df['x'], df['y'])
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:




