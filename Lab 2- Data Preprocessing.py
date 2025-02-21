#!/usr/bin/env python
# coding: utf-8

# In[15]:


#2 and 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
data = pd.read_csv(url, names = ["Sample Code Number", "Clump Thickness", "Uniformity of Cell Size", " Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"])
data


# In[16]:


#4

data = data.drop(["Sample Code Number"], axis=1)


# In[17]:


#5

data.replace('?', np.nan, inplace=True)


# In[18]:


#6

print(data.isna().sum())


# In[19]:


#7

cleaned_data = data.dropna()
cleaned_data


# In[26]:


# 7) Draw a boxplot to identify outliers (10 points)
plt.figure(figsize=(10, 6))
cleaned_data.boxplot()
plt.title("Boxplot to Identify Outliers")
plt.xticks(rotation=45)
plt.grid(False)
plt.show()


# In[21]:


#9

dups = cleaned_data.duplicated()
count_dups = dups.sum()
print(f"Total Count of Duplicates: {count_dups}")


# In[22]:


#10

cleaned_data = cleaned_data.drop_duplicates()
cleaned_data


# In[30]:


#11)

plt.figure(figsize=(8, 6))
cleaned_data['Clump Thickness'].astype(int).hist(bins=10, edgecolor='black')
plt.title("Histogram of 'Clump Thickness' Distribution")
plt.xlabel("Clump Thickness")
plt.ylabel("Frequency")
plt.grid(False)
plt.show()


# In[34]:


# 12)
bin_ranges = [1, 3, 5, 7, 11]
bin_labels = ['Very Low', 'Low', 'Medium', 'High']

cleaned_data = cleaned_data.copy()

cleaned_data.loc[:, 'Clump Thickness Category'] = pd.cut(
    cleaned_data['Clump Thickness'].astype(int), 
    bins=bin_ranges, 
    labels=bin_labels, 
    include_lowest=True 
)


category_counts = cleaned_data['Clump Thickness Category'].value_counts()
print("\nClump Thickness Categories:\n", category_counts)

#The range of each category is as follows:
# Very Low: 1-2
# Low: 3-4
# Medium: 5-6
# High: 7-10


# In[38]:


#13
sample_data = cleaned_data.sample(frac=0.01, random_state=42, replace=False)
print("\nRandom Sample (1% of Data):\n", sample_data)

