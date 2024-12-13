#!/usr/bin/env python
# coding: utf-8

# # INSY 695 Assignment 1
# Amy Hanvoravongchai - 261214198 
# 
# Nov 2024
# 
# ### Objective
# The objective of this project is to predict the wine quality via the best performing classification model.
# 
# #### Business Objective
# To build a robust predictive model to assess the quality of wine based on its physicochemical properties, enabling winemakers, retailers, and quality control teams to:
# - Optimize wine production processes by identifying key factors affecting quality.
# - Ensure consistent product quality to enhance customer satisfaction.
# - Streamline quality control by automating the evaluation process, reducing manual errors and inefficiencies.
# 
# #### Value Proposition
# - **For Winemakers**: Identify areas of improvement in production to enhance wine quality.
# - **For Retailers**: Categorize wines into quality tiers for better pricing and marketing strategies.
# - **For Quality Control Teams**: Reduce manual testing time and costs while maintaining high-quality standards.
# 
# 
# #### Data Information
# The dataset can be found [here](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
from plotly import express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# ## Data Exploration
# A First Look at the Data

# In[240]:


# Load the dataset
wine = pd.read_csv('winequality-red.csv')
wine.head()


# In[241]:


wine.info()


wine.quality.unique()


# The wine quality scores range from 3 to 8.
# 
# I want to categorize the scores into three buckets: 
# - 3 to 4 will be considered "Low" quality.
# - 5 to 6 will be considered "Medium" quality.
# - 7 to 8 will be considered "High" quality.

# In[249]:


wine = wine.replace({'quality': {
                                    8: 'High',
                                    7: 'High',
                                    6: 'Medium',
                                    5: 'Medium',
                                    4: 'Low',
                                    3: 'Low',
                                }
                    })

wine.head()


# ### Normilization & Feature Engineering: Multicollinearity Handling through PCA

# #### Split and Normalize

# In[250]:


y = wine["quality"]
x = wine.drop("quality", axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, shuffle = True, random_state = 1)

# Standardization
norm = MinMaxScaler(feature_range=(0, 1))
norm.fit(x_train)
x_train = norm.transform(x_train)
x_test = norm.transform(x_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ## Building ML Models
# 
#  Random Forest Classifier


# ### Evaluation of Performance Metrics
# 
# KPIs will include:
# - Accuracy 
# - Precision and Recall
# - F1-Score
# - AUC-ROC
# 


# In[251]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, x_test, y_test, average='weighted'):
    # Get predictions
    y_pred = model.predict(x_test)
    
    # If the model supports predict_proba or decision_function, use it
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(x_test)
    else:
        y_prob = None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average)
    f1 = f1_score(y_test, y_pred, average=average)

    # Handle multi-class AUC-ROC
    if y_prob is not None:
        if len(np.unique(y_test)) > 2:  # Multi-class case
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average=average)
        else:  # Binary classification case
            auc = roc_auc_score(y_test, y_prob[:, 1])  # Use probabilities for positive class
    else:
        auc = None

    # Metrics dictionary
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc
    }

    return metrics



# ### Random Forest

# In[253]:


rfc = RandomForestClassifier(
    n_estimators=50, 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    bootstrap=True, 
    random_state=42,
    class_weight='balanced'
)
rfc.fit(x_train, y_train)

rfc_metrics = evaluate_model(rfc, x_test, y_test, average='weighted')
for metric, value in rfc_metrics.items():
    print(f"{metric}: {value}")


# The best performer is Random Forest

import pickle
with open('./models/model.pkl', 'wb') as file:
    pickle.dump(rfc, file)