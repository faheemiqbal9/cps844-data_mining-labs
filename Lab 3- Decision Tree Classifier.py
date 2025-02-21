#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the packages

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# In[2]:


# 1) (5 points) Read the vertebrate.csv data

df = pd.read_csv(r"C:\Users\Kireh Kaka\Downloads\vertebrate.csv")


# In[3]:


# 2) (15 points) The number of records is limited. Convert the data into a binary classification: mammals versus non-mammals
# Hint: ['fishes','birds','amphibians','reptiles'] are considered 'non-mammals'

df['Binary_Class'] = df['Class'].apply(lambda x:1 if x.lower() == 'mammals' else 0)
df.head(10)


# In[10]:


# 3) (15 points) We want to classify animals based on the attributes: Warm-blooded,Gives Birth,Aquatic Creature,
#Aerial Creature,Has Legs,Hibernates
# For training, keep only the attributes of interest, and seperate the target class from the class attributes

attributes = ['Warm-blooded','Gives Birth', 'Aquatic Creature', 'Has Legs', 'Hibernates']
df_cleaned = df[attributes]

df_cleaned.head(10)


# In[12]:


# 4) (10 points) Create a decision tree classifier object. The impurity measure should be based on entropy. Constrain the generated tree with a maximum depth of 3
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)

x = df_cleaned[attributes]
y = df['Binary_Class']


# In[14]:


# 5) (10 points) Train the classifier

clf.fit(x, y)


# In[17]:


# 6) (25 points) Suppose we have the following data
testData = [['lizard',0,0,0,0,1,1,'non-mammals'],
           ['monotreme',1,0,0,0,1,1,'mammals'],
           ['dove',1,0,0,1,1,0,'non-mammals'],
           ['whale',1,1,1,0,0,0,'mammals']]
testData = pd.DataFrame(testData, columns=['name', 'Warm-blooded', 'Gives Birth', 'Aquatic Creature',
                                           'Aerial Creature', 'Has Legs', 'Hibernates', 'class'])


# Prepare the test data and apply the decision tree to classify the test records.
# Extract the class attributes and target class from 'testData'

# Hint: The classifier should correctly label the vertabrae of 'testData' except for the monotreme

x_test = testData[attributes] 
y_pred = clf.predict(x_test)
testData['predicted_class'] = y_pred


# In[18]:


# 7) (10 points) Compute and print out the accuracy of the classifier on 'testData'
testData['binary_class'] = testData['class'].apply(lambda x: 1 if x == 'mammals' else 0)

accuracy = accuracy_score(testData['binary_class'], testData['predicted_class'])
print(f"Accuracy on testData: {accuracy:.2f}")


# In[19]:


# 8) (10 points) Plot your decision tree

plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=attributes, class_names=['Non-Mammal', 'Mammal'], rounded=True)
plt.show()






